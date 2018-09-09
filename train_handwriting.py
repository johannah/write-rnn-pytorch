import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os, sys
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from IPython import embed
import shutil
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
from torch.autograd import Variable # storing data while learning
from handwriting_lstm import mdnLSTM
from utils import save_checkpoint
from dutils import DataLoader
rdn = np.random.RandomState(33)
# TODO one-hot the action space?

torch.manual_seed(139)
# David's blog post:
# https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
# target is nput data shifted by one time step
# input data should be - timestep, batchsize, features!

def train(x, y, validation=False):
    optim.zero_grad()
    bs = x.shape[1]
    h1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c1_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    h2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    c2_tm1 = Variable(torch.zeros((bs, hidden_size))).to(DEVICE)
    outputs = []
    # one batch of x
    for i in np.arange(0,x.shape[0]):
        xin = x[i]
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(xin, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        outputs+=[output]
    y_pred = torch.stack(outputs, 0)
    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos = lstm.get_mixture_coef(y_pred)
    loss = lstm.get_lossfunc(out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos, y[:,:,0][:,:,None], y[:,:,1][:,:,None], y[:,:,2][:,:,None])
    if not validation:
        loss.backward()
        for p in lstm.parameters():
            p.grad.data.clamp_(min=-grad_clip,max=grad_clip)
        optim.step()
    rloss = loss.cpu().data.numpy()
    return y_pred, rloss

def loop(data_loader, num_epochs=1000, save_every=1000, train_losses=[], test_losses=[], train_cnts=[], test_cnts=[]):
    print("starting training loop for data with %s batches"%data_loader.num_batches)
    st = time.time()
    if len(train_losses):
        last_save = train_cnts[-1]
        cnt = train_cnts[-1]
    else:
        last_save = 0
        cnt = 0
    v_x, v_y = data_loader.validation_data()
    v_x = Variable(torch.FloatTensor(np.swapaxes(v_x,1,0))).to(DEVICE)
    v_y = Variable(torch.FloatTensor(np.swapaxes(v_y,1,0))).to(DEVICE)

    print("DUMMMY Validation")
    for i in range(v_x.shape[1]):
        v_x[:,i] = v_x[:,0]
        v_y[:,i] = v_y[:,0]

    for e in range(num_epochs):
        ecnt = 0
        tst = round((time.time()-st)/60., 0)
        if not e%1 and e>0:
            print("starting epoch %s, %s mins, loss %s, seen %s, last save at %s" %(e, tst, train_losses[-1], cnt, last_save))
        batch_loss = []
        for b in range(data_loader.num_batches):
            x, y = data_loader.next_batch()
            x = Variable(torch.FloatTensor(np.swapaxes(x,1,0))).to(DEVICE)
            y = Variable(torch.FloatTensor(np.swapaxes(y,1,0))).to(DEVICE)
            #y_pred, loss = train(x.to(DEVICE),y.to(DEVICE),validation=False)
            y_pred, loss = train(v_x, v_y, validation=False)
            print('DUMMY test loss', loss)
            train_cnts.append(cnt)
            train_losses.append(loss)

            if cnt-last_save >= save_every:
                last_save = cnt
                # find test loss
                valy_pred, val_mean_loss = train(v_x,v_y,validation=True)
                test_losses.append(val_mean_loss)
                test_cnts.append(cnt)
                print('epoch: {} saving after example {} train loss {} test loss {}'.format(e,cnt,loss,val_mean_loss))
                state = {
                        'train_cnts':train_cnts,
                        'train_losses':train_losses,
                        'test_cnts':  test_cnts,
                        'test_losses':test_losses,
                        'state_dict':lstm.state_dict(),
                        'optimizer':optim.state_dict(),
                         }
                filename = os.path.join(savedir, '%s_%015d.pkl'%(model_save_name,cnt))
                save_checkpoint(state, filename=filename)

            cnt+= x.shape[1]
            ecnt+= x.shape[1]

def valid_loop(function, xvar, yvar):
    aloss = []
    cnt = 0
    vdshape = (yvar.shape[0], yvar.shape[1], yvar.shape[2])
    trues = np.zeros(vdshape)
    predicts = np.zeros(vdshape)
    batch_loss = []
    ecnt = 0
    for bst in np.arange(0, (xvar.shape[1]-batch_size)+1, batch_size, dtype=np.int):
        xd = xvar[:,bst:bst+batch_size]
        yd = yvar[:,bst:bst+batch_size]
        y_pred, losses = function(lstm, hidden_size, DEVICE, mse_loss, xd, yd)
        predicts[:,bst:bst+batch_size,:] = y_pred.detach().numpy()
        trues[:,bst:bst+batch_size,:] = yd.detach().numpy()
        cnt+=batch_size
        ecnt+=batch_size
        batch_loss.extend(losses)
    # get leftovers
    num_left = xvar.shape[1]-ecnt
    if num_left:
        xd = xvar[:,ecnt:]
        yd = yvar[:,ecnt:]
        y_pred, losses = function(lstm, hidden_size, DEVICE, mse_loss, xd, yd)
        predicts[:,ecnt:,:] = y_pred.detach().numpy()
        trues[:,ecnt:,:] = yd.detach().numpy()
        cnt+=num_left
        batch_loss.extend(losses)
    return trues, predicts, batch_loss

def plot_results(cnt, vx_tensor, vy_tensor, name='test'):
    # check that feature array for offset is correct from training set
    print("predicting results for %s" %name)
    tf_trues, tf_predicts, tf_batch_loss = valid_loop(teacher_force_predict, vx_tensor, vy_tensor)
    trues, predicts, pbatch_loss = valid_loop(predict, vx_tensor, vy_tensor)
    print("plotting results for %s" %name)
    if not os.path.exists(img_savedir):
        os.makedirs(img_savedir)
    for e in range(trues.shape[1]):
        filename = os.path.join(img_savedir, '%s_%s_%05d.png'%(model_save_name.replace('.pkl',''),name,e))
        plot_traces(trues[:,e], tf_predicts[:,e], predicts[:,e], filename)


if __name__ == '__main__':
    import argparse

    learning_rate = 0.0001
    batch_size = 32
    seq_length = 300
    hidden_size = 1024
    savedir = 'models'
    number_mixtures = 20
    grad_clip = 10
    data_scale = 20
    input_size = 3
    train_losses, test_losses, train_cnts, test_cnts = [], [], [], []

    img_savedir = 'predictions'
    cnt = 0
    default_model_loadname = 'models/model_000000003700572.pkl'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-po', '--plot', action='store_true', default=False)
    parser.add_argument('-m', '--model_loadname', default=default_model_loadname)
    parser.add_argument('-ne', '--num_epochs',default=300, help='num epochs to train')
    parser.add_argument('-se', '--save_every',default=10000, help='how often in epochs to save training model')
    parser.add_argument('--limit', default=-1, type=int, help='limit training data to reduce convergence time')
    parser.add_argument('-l', '--load', action='store_true', default=False, help='load model to continue training or to generate. model path is specified with -m')
    parser.add_argument('-v', '--validate', action='store_true', default=False, help='test results')

    args = parser.parse_args()

    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    save_every = args.save_every
    data_loader = DataLoader(batch_size, seq_length, data_scale)

    v_x, v_y = data_loader.validation_data()
    lstm = mdnLSTM(input_size=input_size, hidden_size=hidden_size, number_mixtures=number_mixtures).to(DEVICE)
    optim = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    model_save_name = 'model'
    if args.limit != -1:
        model_save_name += "_limit_%04d"%args.limit

    if args.load:
        if not os.path.exists(args.model_loadname):
            print("load model: %s does not exist"%args.model_loadname)
            sys.exit()
        else:
            print("loading %s" %args.model_loadname)
            lstm_dict = torch.load(args.model_loadname)
            lstm.load_state_dict(lstm_dict['state_dict'])
            optim.load_state_dict(lstm_dict['optimizer'])
            train_cnts = lstm_dict['train_cnts']
            train_losses = lstm_dict['train_losses']
            test_cnts = lstm_dict['test_cnts']
            test_losses = lstm_dict['test_losses']


    if not args.plot:
        loop(data_loader, save_every=save_every, num_epochs=args.num_epochs, train_losses=[], test_losses=[], train_cnts=[], test_cnts=[])

    #else:
    #    model_save_name = os.path.split(model_load_path)[1]
    #    plot_results(cnt, valid_x_tensor[:,:batch_size], valid_y_tensor[:,:batch_size], name='test')
    #    plot_results(cnt,       x_tensor[:,:batch_size],       y_tensor[:,:batch_size], name='train')

    embed()


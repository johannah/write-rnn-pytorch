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
from utils import save_checkpoint, plot_strokes, get_dummy_data
from dutils import DataLoader
rdn = np.random.RandomState(33)
# TODO one-hot the action space?

torch.manual_seed(139)
# David's blog post:
# https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
# input data should be - timestep, batchsize, features!
# Sampling example from tfbldr kkastner
# https://github.com/kastnerkyle/tfbldr/blob/master/examples/handwriting/generate_handwriting.py

def sample_2d(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1*std1, std1*std2*rho],
                    [std1*std2*rho, std2*std2]])
    mean = np.array([mu1,mu2])
    x,y = rdn.multivariate_normal(mean, cov, 1)[0]
    end = rdn.binomial(1,e)
    return np.array([x,y,end])

def get_pi_idx(x, pdf):
    N = pdf.shape[0]
    accumulate = 0
    for i in range(N):
        accumulate+=pdf[i]
        if accumulate>=x:
            return i
    print("error sampling")
    return -1

def predict(x, h1_tm1, c1_tm1, h2_tm1, c2_tm1, batch_num=0, use_center=True):
    # one batch of x
    output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(x, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
    # out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos
    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos = lstm.get_mixture_coef(output)
    mso = (out_pi.cpu().data.numpy(),
           out_mu1.cpu().data.numpy(), out_mu2.cpu().data.numpy(),
           out_sigma1.cpu().data.numpy(), out_sigma2.cpu().data.numpy(),
           out_corr.cpu().data.numpy(), out_eos.cpu().data.numpy())
    pi, mu1, mu2, sigma1, sigma2, corr, eos = mso
    # choose mixture
    bn = batch_num
    idx = rdn.choice(np.arange(pi.shape[1]), p=pi[bn])
    pred = sample_2d(eos[bn], mu1[bn,idx], mu2[bn,idx], sigma1[bn,idx], sigma2[bn,idx], corr[bn,idx])
    if use_center:
        pred = np.array([mu1[bn,idx], mu2[bn,idx], 0])
    return pred, h1_tm1, c1_tm1, h2_tm1, c2_tm1

def generate(modelname, num=300,  data_scale=20, teacher_force_predict=True, use_center=False):
    bn = 0
    batch_size = 1
    assert(bn<batch_size)
    h1_tm1 = Variable(torch.zeros((batch_size, hidden_size))).to(DEVICE)
    c1_tm1 = Variable(torch.zeros((batch_size, hidden_size))).to(DEVICE)
    h2_tm1 = Variable(torch.zeros((batch_size, hidden_size))).to(DEVICE)
    c2_tm1 = Variable(torch.zeros((batch_size, hidden_size))).to(DEVICE)

    if teacher_force_predict:
        data_loader = DataLoader(batch_size, seq_length, data_scale)
        x,y = data_loader.validation_data()
        x = Variable(torch.FloatTensor(np.swapaxes(x,1,0))).to(DEVICE)
        num=x.shape[0]
        last_x = x[0,:bn+1,:]
    else:
        last_x = Variable(torch.FloatTensor(np.zeros((batch_size,3)))).to(DEVICE)
    last_x[:,2] = 1.0 # make new stroke
    strokes = np.zeros((num,3), dtype=np.float32)
    strokes[0] = last_x
    for i in range(num-1):
        pred, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = predict(last_x, h1_tm1, c1_tm1, h2_tm1, c2_tm1, use_center=use_center)
        strokes[i+1] = pred
        if teacher_force_predict:
            # override
            last_x = x[i+1,:bn+1,:]
        else:
            last_x = torch.FloatTensor(pred[None,:])

    strokes[:,:2]*=data_scale
    base = '_gen'
    if use_center:
        base = base+'_center'
    if teacher_force_predict:
        fname = os.path.join(modelname.replace('.pkl', base+'_tf.png'))
        print("plotting teacher force generation: %s" %fname)
        xtrue = x[:,bn].cpu().data.numpy()
        strokes[:,2] = xtrue[:,2]
        plot_strokes(xtrue, strokes, name=fname)
    else:
        fname = os.path.join(modelname.replace('.pkl', base+'.png'))
        print("plotting generation: %s" %fname)
        plot_strokes(strokes, strokes*0.0, name=fname)
    embed()

if __name__ == '__main__':
    import argparse
    batch_size = 32
    seq_length = 300
    hidden_size = 1024
    savedir = 'models'
    number_mixtures = 20
    data_scale = 1
    input_size = 3
    train_losses, test_losses, train_cnts, test_cnts = [], [], [], []

    img_savedir = 'predictions'
    cnt = 0
    default_model_loadname = 'models/model_000000000030048.cpkl'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(img_savedir):
        os.makedirs(img_savedir)
    parser = argparse.ArgumentParser()
    parser.add_argument('model_loadname', default=default_model_loadname)
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-uc', '--use_center', action='store_true', default=False, help='use means instead of sampling')
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    parser.add_argument('-n', '--num',default=300, help='length of data to generate')
    parser.add_argument('--num_plot', default=10, type=int, help='number of examples from training and test to plot')

    args = parser.parse_args()

    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'


    lstm = mdnLSTM(input_size=input_size, hidden_size=hidden_size, number_mixtures=number_mixtures).to(DEVICE)
    model_save_name = 'model'
    if not os.path.exists(args.model_loadname):
        print("load model: %s does not exist"%args.model_loadname)
        sys.exit()
    else:
        print("loading %s" %args.model_loadname)
        lstm_dict = torch.load(args.model_loadname)
        lstm.load_state_dict(lstm_dict['state_dict'])
        train_cnts = lstm_dict['train_cnts']
        train_losses = lstm_dict['train_losses']
        test_cnts = lstm_dict['test_cnts']
        test_losses = lstm_dict['test_losses']

    generate(args.model_loadname, num=args.num,  data_scale=data_scale, teacher_force_predict=args.teacher_force, use_center=args.use_center)
    embed()


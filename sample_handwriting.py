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

def teacher_force_predict(x, validation=False):
    bs = x.shape[1]
    ts = x.shape[0]
    number_mixtures = 20
    out_shape = (ts,bs,number_mixtures)
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
    y_pred_flat = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])
    # out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos
    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos = lstm.get_mixture_coef(y_pred_flat)
    mso = (out_pi.reshape(out_shape).cpu().data.numpy(),
           out_mu1.reshape(out_shape).cpu().data.numpy(), out_mu2.reshape(out_shape).cpu().data.numpy(),
           out_sigma1.reshape(out_shape).cpu().data.numpy(), out_sigma2.reshape(out_shape).cpu().data.numpy(),
           out_corr.reshape(out_shape).cpu().data.numpy(), out_eos.reshape(ts, bs, 1).cpu().data.numpy())
    return y_pred, mso

def generate_teacher_force(modelname, data_scale=20, dummy=False):
    # initialize as start of stroke
    #    prev_x = Variable(torch.FloatTensor(np.zeros((1,1,3)))).to(DEVICE)
    x,y = data_loader.validation_data()
    x = Variable(torch.FloatTensor(np.swapaxes(x,1,0))).to(DEVICE)
    y = Variable(torch.FloatTensor(np.swapaxes(y,1,0))).to(DEVICE)
    if dummy:
        print("GENERATING FROM DUMMY DATA")
        x,y = get_dummy_data(x,y)
    num=x.shape[0]
    y_pred, mxt = teacher_force_predict(x[:,:1])
    pi, mu1, mu2, sigma1, sigma2, corr, eos = mxt
    strokes = np.zeros((num,3), dtype=np.float32)
    bn = 0
    for i in range(num):
        idx = get_pi_idx(rdn.rand(), pi[i,bn])
        strokes[i] = sample_2d(eos[i,0], mu1[i,0,idx], mu2[i,0,idx], sigma1[i,0,idx], sigma2[i,0,idx], corr[i,0,idx])
    strokes[:,:2]*=data_scale
    fname = os.path.join(img_savedir, modelname.replace('.pkl', '_tf.png'))
    plot_strokes(x[:,0].cpu().data.numpy(),strokes, name=fname)

if __name__ == '__main__':
    import argparse

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
    default_model_loadname = 'models/model_000000000030048.cpkl'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(img_savedir):
        os.makedirs(img_savedir)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--dummy', action='store_true', default=False)
    parser.add_argument('-m', '--model_loadname', default=default_model_loadname)
    parser.add_argument('-ne', '--num_epochs',default=300, help='num epochs to train')
    parser.add_argument('-se', '--save_every',default=10000, help='how often in epochs to save training model')
    parser.add_argument('--num_plot', default=10, type=int, help='number of examples from training and test to plot')

    args = parser.parse_args()

    if args.cuda:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    save_every = args.save_every
    data_loader = DataLoader(batch_size, seq_length, data_scale)

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

    modelname = os.path.split(args.model_loadname)[1]
    plot_basename = os.path.join(img_savedir, modelname).replace('.pkl', '')
    generate_teacher_force(modelname, data_scale=20, dummy=args.dummy)
    #generate(args.num_plot, train_cnts[-1], train_losses[-1], plot_basename)
    embed()


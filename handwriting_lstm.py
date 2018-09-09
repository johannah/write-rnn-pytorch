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
rdn = np.random.RandomState(33)
# TODO one-hot the action space?
torch.manual_seed(139)

class mdnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=1024, number_mixtures=20):
        super(mdnLSTM, self).__init__()
        self.number_mixtures = number_mixtures
        # one extra output for pen up
        # multiple by 3 for each of the two output
        self.output_size = 1+self.number_mixtures*6
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, self.output_size)

    def forward(self, xt, h1_tm1, c1_tm1, h2_tm1, c2_tm1):
        h1_t, c1_t = self.lstm1(xt, (h1_tm1, c1_tm1))
        h2_t, c2_t = self.lstm2(h1_t, (h2_tm1, c2_tm1))
        output = self.linear(h2_t)
        return output, h1_t, c1_t, h2_t, c2_t

    def get_mixture_coef(self, output):
        # split of data into pi,sigma,mu for each feature dimension
        z = output
        # z_os is end of stroke signal
        out_eos = torch.sigmoid(z[:,:1])
        # split into six pieces
        z_pi, out_mu1, out_mu2, z_sigma1, z_sigma2, z_corr = torch.split(z[:,:,1:], self.number_mixtures, dim=2)
        # softmax the pis
        max_pi,_ = torch.max(z_pi, dim=2, keepdim=True)
        exp_pi = torch.exp(z_pi-max_pi)
        out_pi = exp_pi/torch.sum(exp_pi, dim=2, keepdim=True)
        out_sigma1 = torch.exp(z_sigma1)
        out_sigma2 = torch.exp(z_sigma2)
        out_corr = torch.tanh(z_corr)
        return out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, out_eos

    def pt_2d_normal(self, x1, x2, mu1, mu2, sigma1, sigma2, rho):
        norm1 = x1-mu1
        norm2 = x2-mu2
        s1s2 = sigma1*sigma2
        z = (norm1/sigma1)**2 + (norm2/sigma2)**2 - 2*((rho*(norm1*norm2))/s1s2)
        negRho = 1-(rho**2)
        result = torch.exp(-z/(2*negRho))
        denom = 2*np.pi*(s1s2*torch.sqrt(negRho))
        result = result/denom
        return result

    def get_lossfunc(self, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos, x1_data, x2_data, eos_data):
        result0 = self.pt_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
        epsilon = 1e-20
        result1 = torch.sum(result0*z_pi, dim=2, keepdim=True)
        result1 = -torch.log(result1+epsilon)
        result2 = -torch.log((z_eos*eos_data) + (1-z_eos)*(1-eos_data))
        result = result1+result2
        result = torch.sum(result)
        return result

    def sample(self, sess, num=1200):
        def get_pi_idx(x, pdf):
            N = pdf.shape[0]
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i
            print('error with sampling ensemble')
            return -1

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
            mean = [mu1, mu2]
            cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
        # TODO
        #prev_state = sess.run(self.cell.zero_state(1, tf.float32))
        #strokes = np.zeros((num, 3), dtype=np.float32)
        #mixture_params = []

        #for i in range(num):

        #    feed = {self.input_data: prev_x, self.state_in: prev_state}

        #    [o_pi,
        #     o_mu1,
        #     o_mu2,
        #     o_sigma1,
        #     o_sigma2,
        #     o_corr,
        #     o_eos,
        #     next_state] = sess.run([self.pi,
        #                             self.mu1,
        #                             self.mu2,
        #                             self.sigma1,
        #                             self.sigma2,
        #                             self.corr,
        #                             self.eos,
        #                             self.state_out],
        #                            feed)

        #    idx = get_pi_idx(random.random(), o_pi[0])

        #    eos = 1 if random.random() < o_eos[0][0] else 0

        #    next_x1, next_x2 = sample_gaussian_2d(
        #        o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

        #    strokes[i, :] = [next_x1, next_x2, eos]

        #    params = [
        #        o_pi[0],
        #        o_mu1[0],
        #        o_mu2[0],
        #        o_sigma1[0],
        #        o_sigma2[0],
        #        o_corr[0],
        #        o_eos[0]]
        #    mixture_params.append(params)

        #    prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        #    prev_x[0][0] = np.array([next_x1, next_x2, eos], dtype=np.float32)
        #    prev_state = next_state

        #strokes[:, 0:2] *= self.args.data_scale
        #return strokes, mixture_params





import torch
import torch.nn as nn
import numpy as np
import pytorch_msssim

ssim = pytorch_msssim.ms_ssim
theta_range = np.linspace(0,np.pi/2,64)
phi_range = np.linspace(0,np.pi,64)
sincos = np.outer(np.sin(theta_range),np.cos(phi_range))
sinsin = np.outer(np.sin(theta_range),np.sin(phi_range))
p = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]])*2*0.45*np.pi
for i in range(6):
    tmp = np.exp(-1j*(sincos*p[i,0] + sinsin*p[i,1]))
    if i == 0:
        factor = tmp.reshape(1,64,64)
    else:
        factor = np.concatenate([factor,tmp.reshape(1,64,64)],0)

def array_gain_calc(G,pos): 
    G = G.squeeze()
    if len(G.shape) == 2:
        G = G.unsqueeze(0)
    n_ants = len(pos)
    max_values = torch.max(G.view(n_ants,-1),dim = 1).values
    for i in range(G.shape[0]):
        G[i,:,:] = G[i,:,:] - max_values[i]
    G = 10**G
    num = G.shape[0]
    G = G.numpy()
    t_factor = torch.max(max_values).item()
    max_values = max_values -t_factor
    second_factor = 10**(max_values)
    second_factor = second_factor.numpy()
    for i in range(num):
        if i == 0:
            out = factor[pos[i],:,:] * G[i,:,:] * second_factor[i]
        else:
            out = out + factor[pos[i],:,:] * G[i,:,:] * second_factor[i]
    out = np.abs(out)
    out = 10*np.log10(out + 1e-30) + t_factor
    return torch.tensor(out)


def array_gain_calc_torch(G,factor): 
    G = G.squeeze()
    max_values = torch.max(G.view(6,-1),dim = 1).values
    for i in range(G.shape[0]):
        G[i,:,:] = G[i,:,:] - max_values[i]
    G = 10**G
    num = G.shape[0]
    t_factor = torch.max(max_values).item()
    max_values = max_values -t_factor
    second_factor = 10**(max_values)
    for i in range(num):
        if i == 0:
            out = factor[i,:,:] * G[i,:,:] * second_factor[i]
        else:
            out = out + factor[i,:,:] * G[i,:,:] * second_factor[i]
    out = torch.abs(out)
    out = 10*torch.log10(out + 1e-30) + t_factor
    return out

sigmoid = nn.Sigmoid()

def clapped_log(source):
    x = torch.log2(source).clamp_min_(-100)
    return x

def half_bce(source,target):
    loss = -(source.squeeze())*clapped_log(target.squeeze())
    loss = torch.mean(loss)
    return loss
def recall_loss(constraints,decisions):
    flatten_decision = sigmoid(torch.sum(decisions,0))
    return half_bce(constraints,flatten_decision)

def ssim_loss(source,target):
    ag_s = array_gain_calc(source)
    loss_1 = -ssim(X = ag_s.unsqueeze(0).unsqueeze(0),Y = target.unsqueeze(0).unsqueeze(0),data_range = 182,win_size = 3)
    return loss_1

crossentropyloss = nn.CrossEntropyLoss()
def structural_loss(geometry,decision,flag=False):
    decision = decision.squeeze()
    batch_size = decision.shape[0] 
    for i in range(2):
        dec = decision[:,i,:,:,:].squeeze().reshape(-1,2)
        gem = geometry[:,i,:,:].squeeze().reshape(-1)
        if i == 0:
            gem_loss = crossentropyloss(dec,gem)
        gem_loss += crossentropyloss(dec,gem)
    loss =  gem_loss
    return loss

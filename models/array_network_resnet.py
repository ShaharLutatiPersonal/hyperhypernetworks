import torch
import torch.nn as nn
import numpy as np
from loss import array_gain_calc as ag_calc
import transformer
import position_encoding
import resnet
import math
fc = [dict(),dict(),dict(),dict()]
fc[0]['shape'] = [3,32]
fc[1]['shape'] = [32,64]
fc[2]['shape'] = [64,64]
fc[3]['shape'] = [64,3]
cnt = 0

for fcc in fc:
    shape = fcc['shape']
    cnt += shape[0]*shape[1] + 2*shape[1]
print(cnt)
class resnet_hyperhyper_variant(nn.Module):
    def __init__(self):
        super(resnet_hyperhyper_variant,self).__init__()
        self.hyper_net = hypernetwork_constraints()
        self.sizes = [72,2304]
        self.backbone_0 =  resnet.ResNetBasicBlock(8,16)
        self.backbone_1 =  nn.Sequential(resnet.ResNetBasicBlock(16,1),nn.Conv2d(1,1,3),nn.BatchNorm2d(1),nn.Conv2d(1,1,3))
        self.activation = nn.ELU()
        self.relu = nn.ReLU()
        self.linear_0 = nn.Linear(3136*1+3,3136*1+3)
        self.linear_class_hyper = nn.Linear(3136*1+3,cnt)
        self.sigmoid = nn.Identity()
        self.init_weight_network(self.linear_class_hyper.weight,64)
    def forward(self, inputs,gcoord,constraint,s):
        weights = self.hyper_net(constraint).squeeze()
        weights_conv_0 = weights[:self.sizes[0]]
        weights_conv_0 = weights_conv_0.view(8,1,3,3)
        weights_conv_1 = weights[self.sizes[0]:self.sizes[0]+self.sizes[1]]
        weights_conv_1 = weights_conv_1.view(16,16,3,3)
        x = nn.functional.conv2d(inputs,weights_conv_0) # Layer index 1
        x = self.activation(x)
        x = self.backbone_0(x)# Layer index 2-8
        x = nn.functional.conv2d(x,weights_conv_1) # Layer index 9
        x = self.backbone_1(x) # Layer index 10-18
        x = x.view(1,-1)
        x = torch.cat((x,s),dim=1)
        x = self.linear_0(x) # Layer index 19
        linear_weights = self.activation(self.linear_class_hyper(x)).squeeze() # Layer index 20
        tot_before = 0
        output = gcoord
        for i,fcc in enumerate(fc):
            shape = fcc['shape']
            h,w = shape[0],shape[1]
            lin = linear_weights[tot_before:tot_before+h*w+2*w]
            linear = lin[:h*w].view(w,h)
            bias = lin[h*w:h*w+w]
            scale = lin[h*w+w:h*w+2*w]
            tot_before += h*w+w
            if i == (len(fc) -1):
                output = self.scaled_linear(w = linear,s = scale,b = bias,input = output)
            else:
                output = self.relu(self.scaled_linear(w = linear,s = scale,b = bias,input =output))
        return self.sigmoid(output)

    def scaled_linear(self,w,s,b,input):
        input = input.matmul(w.t())
        input = input * s + b
        return input

    def init_weight_network(self,w,fan_out_dj):
        with torch.no_grad():
            fan_in = w.shape[0]
            a = torch.sqrt(torch.tensor(3/(fan_in*fan_out_dj)))
            return w.uniform_(-a,a)


    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if (name not in own_state):
                    continue
            if (own_state[name].shape != param.shape):
                continue
            if isinstance(param, torch.nn.Parameter):
                print(name)
                # backwards compatibility for serialized parameters
                param = param.data
            print(name)
            own_state[name].copy_(param)


class hypernetwork_constraints(nn.Module):
    def __init__(self):
        super(hypernetwork_constraints,self).__init__()
        hidden_dim = 128
        self.backbone = nn.Sequential(nn.Conv2d(1,8,5),nn.MaxPool2d(2),nn.ELU(),nn.Conv2d(8,64,3),nn.MaxPool2d(3),
        nn.ELU(), nn.Conv2d(64,64,3),nn.MaxPool2d(3),nn.ELU(),nn.Conv2d(64,hidden_dim,3),nn.ELU(), nn.Conv2d(hidden_dim,hidden_dim,1))
        self.linear_0 = nn.Linear(in_features = 7*4*128, out_features = hidden_dim*3)
        self.linear_1 = nn.Linear(in_features = hidden_dim*3,out_features = 2376)
        self.activation = nn.ELU()
        self.init_hyperhyper(self.linear_1.weight,100*100*0.209)
        nn.init.kaiming_uniform_(self.linear_0.weight)
    def forward(self,input):
        out = self.backbone(100*(1-input))
        #print(out.shape)
        out = self.linear_0(out.view(1,-1))
        out = self.activation(out)
        out = self.linear_1(out)
        return out
    
    def init_hyperhyper(self,w,varc):
        with torch.no_grad():
            fan_out = w.shape[1]
            a = torch.sqrt(torch.tensor(3*2/(fan_out*varc)))
            return w.uniform_(-a,a)

import resnet
class hyper_resnet(nn.Module):
    def __init__(self):
        super(hyper_resnet,self).__init__()
        
        self.relu = nn.ELU()
        self.directivity_conv = nn.Sequential(resnet.ResNetBasicBlock(64,16),
        resnet.ResNetBasicBlock(16,16), resnet.ResNetBasicBlock(16,1))

        
    def forward(self,input):
        directivity = input
        directivity = self.directivity_conv(directivity)
        concat = directivity.view(1,-1)
        return concat
import torch
import torch.nn as nn
import numpy as np
from loss import array_gain_calc as ag_calc
import math
fc = [dict(),dict(),dict(),dict()]
fc[0]['shape'] = [3,32]
fc[1]['shape'] = [32,64]
fc[2]['shape'] = [64,64]
fc[3]['shape'] = [64,2]
cnt = 0

for fcc in fc:
    shape = fcc['shape']
    cnt += shape[0]*shape[1] + 2*shape[1]
print(cnt)
class transformer_no_hyperhypernet(nn.Module):
    def __init__(self,gem = 64, num_classes = 2, hidden_dim = 128, nheads = 8,num_encoder_layers = 3, num_decoder_layers = 1):
        super(transformer_no_hyperhypernet,self).__init__()
        self.sizes = [72,4608,3136]
        self.backbone_m1 = nn.Sequential(nn.Conv2d(1,8,3),nn.ELU(),nn.Conv2d(8,64,3),nn.ELU())
        self.backbone_0 =  nn.Sequential(nn.Conv2d(64,64,3),nn.ELU(), nn.Conv2d(64,hidden_dim,3),nn.ELU(), nn.Conv2d(hidden_dim,hidden_dim,1))
        self.activation = nn.ELU()
        self.relu = nn.ReLU()
        transformer_encod = nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=4,dim_feedforward=1024,dropout=.1)
        self.transformer = nn.TransformerEncoder(transformer_encod,4)
        self.conv11 = nn.Conv1d(3136,1,1)
        self.bn_0 = nn.GroupNorm(1,1)
        self.bn_2 = nn.GroupNorm(1,131)
        self.bn_3 = nn.GroupNorm(1,128)
        self.linear_0 = nn.Linear(hidden_dim*1+3,hidden_dim*1)
        self.linear_class_hyper = nn.Linear(hidden_dim*1,cnt)
        self.pos = self.positionalencoding2d(d_model=hidden_dim, height=56, width=56).to('cuda')
        self.hidden_dim = hidden_dim
        self.gem = gem
        self.sigmoid = nn.Identity()
        self.init_weight_network(self.linear_class_hyper.weight,64)
    def forward(self, inputs,gcoord,constraint,s):
        hidden_dim = self.hidden_dim
        x = self.backbone_m1(inputs)
        x = self.activation(x)
        x = self.backbone_0(x)
        pos = self.pos
        m = pos +x.squeeze()
        m = m.permute(2,1,0)
        m = m.contiguous().view(56**2,1,-1)
        h = self.transformer(m)
        h = self.conv11(h.permute(1,0,2))
        h = self.bn_0(h)
        h = h.contiguous().view(1,-1)
        h = torch.cat((h,s),1)
        h = self.bn_1(h)
        h = self.activation(self.linear_0(h))
        h = self.bn_3(h)
        linear_weights = self.activation(self.linear_class_hyper(h)).squeeze()
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
            fan_in = w.shape[1]
            a = torch.sqrt(torch.tensor(3/(fan_in*fan_out_dj)))
            return w.uniform_(-a,a)

    def positionalencoding2d(self,d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe

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
        self.linear_1 = nn.Linear(in_features = hidden_dim*3,out_features = 7816)
        self.activation = nn.ELU()

    def forward(self,input):
        out = self.backbone(100*(1-input))
        #print(out.shape)
        out = self.linear_0(out.view(1,-1))
        out = self.activation(out)
        out = self.linear_1(out)
        return out


fc[0]['shape'] = [3,32]
fc[1]['shape'] = [32,64]
fc[2]['shape'] = [64,64]
fc[3]['shape'] = [64,2]


class transformer_no_hypernet(nn.Module):
    def __init__(self,gem = 64, num_classes = 2, hidden_dim = 128, nheads = 8,num_encoder_layers = 3, num_decoder_layers = 1):
        super(transformer_no_hypernet,self).__init__()
        self.sizes = [72,4608,3136]
        self.backbone_m1 = nn.Sequential(nn.Conv2d(1,8,3),nn.ELU(),nn.Conv2d(8,64,3),nn.ELU())
        self.backbone_0 =  nn.Sequential(nn.Conv2d(64,64,3),nn.ELU(), nn.Conv2d(64,hidden_dim,3),nn.ELU(), nn.Conv2d(hidden_dim,hidden_dim,1))
        self.activation = nn.ELU()
        self.relu = nn.ReLU()
        transformer_encod = nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=4,dim_feedforward=1024,dropout=.1)
        self.transformer = nn.TransformerEncoder(transformer_encod,3)
        self.linear_block = nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.last_block = nn.Linear(64,3)
        self.conv11 = nn.Conv1d(3136,1,1)
        self.linear_0 = nn.Linear(hidden_dim*1+3,64)
        self.pos = self.positionalencoding2d(d_model=hidden_dim, height=56, width=56).to('cuda')
        self.hidden_dim = hidden_dim
        self.gem = gem
        self.sigmoid = nn.Identity()
    def forward(self, inputs,gcoord,constraint,s):
        hidden_dim = self.hidden_dim
        x = self.backbone_m1(inputs)
        x = self.activation(x)
        x = self.backbone_0(x)
        pos = self.pos
        m = pos +x.squeeze()
        m = m.permute(2,1,0)
        m = m.contiguous().view(56**2,1,-1)
        h = self.transformer(m)
        h = self.conv11(h.permute(1,0,2))
        h = h.contiguous().view(1,-1)
        h = torch.cat((h,s),1)
        h = self.activation(self.linear_0(h))
        output = gcoord
        output = self.linear_block(output)
        output = self.last_block(output.mul(h))
        return self.sigmoid(output)

    def scaled_linear(self,w,s,b,input):
        input = input.matmul(w.t())
        input = input * s + b
        return input

    def init_weight_network(self,w,fan_out_dj):
        with torch.no_grad():
            fan_in = w.shape[1]
            a = torch.sqrt(torch.tensor(3/(fan_in*fan_out_dj)))
            return w.uniform_(-a,a)

    def positionalencoding2d(self,d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe

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
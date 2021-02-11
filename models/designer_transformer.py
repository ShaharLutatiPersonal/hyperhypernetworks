import torch
import torch.nn as nn
import numpy as np
import cost
import resnet
import transformer
import position_encoding
import math

class designer_network(nn.Module):
    def __init__(self,fc_shape):
        super(designer_network,self).__init__()
        self.hypernet = hypernetwork_constarints(fc_parameters=fc_shape)
        self.relu = nn.ELU()
        self.activation = nn.Sigmoid()
        self.fc_shape = fc_shape
    def forward(self,inputs):
        cord,constraints = inputs
        linear_weights = self.hypernet(constraints).squeeze()
        tot_before = 0
        output = cord
        for i,fc in enumerate(self.fc_shape):
            shape = fc['shape']
            h,w = shape[0],shape[1]
            lin = linear_weights[tot_before:tot_before+h*w+2*w]
            linear = lin[:h*w].view(w,h)
            bias = lin[h*w:h*w+w]
            scale = lin[h*w+w:h*w+2*w]
            tot_before += h*w+w
            if i == (len(self.fc_shape) -1):
                output = self.scaled_linear(w = linear,s = scale,b = bias,input = output)
            else:
                output = self.relu(self.scaled_linear(w = linear,s = scale,b = bias,input =output))
        return output
    def scaled_linear(self,w,s,b,input):
        input = input.matmul(w.t())
        input = input * s + b
        return input




class designer_fine_tuned_network(nn.Module):
    def __init__(self,fc_shape):
        super(designer_fine_tuned_network,self).__init__()
        self.hypernet = hypernetwork_constarints(fc_parameters=fc_shape)
        self.hyper_fine = fine_tuner_hyper(fc_parameters=fc_shape)
        self.relu = nn.ELU()
        self.activation = nn.Sigmoid()
        self.fc_shape = fc_shape
    def forward(self,inputs):
        (cord,constraints),metal_plane = inputs
        linear_weights = self.hyper_fine(self.hypernet(constraints),metal_plane).squeeze()
        tot_before = 0
        output = cord
        for i,fc in enumerate(self.fc_shape):
            shape = fc['shape']
            h,w = shape[0],shape[1]
            lin = linear_weights[tot_before:tot_before+h*w+2*w]
            linear = lin[:h*w].view(w,h)
            bias = lin[h*w:h*w+w]
            scale = lin[h*w+w:h*w+2*w]
            tot_before += h*w+w
            if i == (len(self.fc_shape) -1):
                output = self.scaled_linear(w = linear,s = scale,b = bias,input = output)
            else:
                output = self.relu(self.scaled_linear(w = linear,s = scale,b = bias,input =output))
        return output
    def scaled_linear(self,w,s,b,input):
        input = input.matmul(w.t())
        input = input * s + b
        return input
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                    continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)










import resnet
class hypernetwork_constarints(nn.Module):
    def __init__(self,fc_parameters):
        super(hypernetwork_constarints,self).__init__()
        cnt = 0
        hidden_dim = 128

        for fcc in fc_parameters:
            shape = fcc['shape']
            cnt += shape[0]*shape[1] + 2*shape[1]
        self.backbone_0 =  nn.Sequential(nn.Conv2d(1,8,3),nn.ReLU(),
        nn.Conv2d(8,8,3),nn.ReLU(),nn.Conv2d(8,8,3),nn.ReLU(), nn.Conv2d(8,hidden_dim,3),nn.ReLU(), nn.Conv2d(hidden_dim,hidden_dim,1))
        self.activation = nn.ELU()
        self.relu = nn.ReLU()
        transformer_encod = nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=4,dim_feedforward=2048,dropout=0.1)
        self.transformer = nn.TransformerEncoder(transformer_encod,4)
        self.conv11 = nn.Sequential(nn.Conv1d(3136,1,1))
        self.hyp = nn.Identity()#nn.Linear(hidden_dim,hidden_dim)
        self.linear_class_hyper = nn.Linear(hidden_dim*1 + 3,cnt)
        self.pos = self.positionalencoding2d(d_model=hidden_dim, height=56, width=56).to('cuda')
        self.hidden_dim = hidden_dim
        self.gem = 64
        self.sigmoid = nn.Identity()
        self.init_weight_network(self.linear_class_hyper.weight,64)
    def forward(self, inputs):
        inputs,s = inputs
        x = self.activation(self.backbone_0(inputs))
        pos = self.pos
        m = pos +x.squeeze()
        m = m.permute(2,1,0)
        m = m.contiguous().view(56**2,1,-1)
        h = self.transformer(m).permute(1,0,2)
        h = self.activation(self.conv11(h))
        h = self.activation(self.hyp(h))
        h = h.contiguous().view(1,-1)
        h = torch.cat((h,s),dim = 1)
        linear_weights = self.activation(self.linear_class_hyper(h)).squeeze()
        return linear_weights
 
    def init_bias_network(self,w):
        with torch.no_grad():
            fan_in = w.shape[0]
            uni = torch.randn(w.shape)
            a = torch.sqrt(torch.tensor(1/fan_in))
            w = uni*a
            return w    

    def init_weight_network(self,w,fan_out_dj):
        with torch.no_grad():
            fan_in = w.shape[1]
            a = torch.sqrt(torch.tensor(3*2/(fan_in*fan_out_dj)))
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
class conv_block(nn.Module):
    def __init__(self,init_shape):
        super(conv_block,self).__init__()
        self.conv0 = nn.Conv2d(in_channels=init_shape[0],out_channels=init_shape[1],kernel_size=init_shape[2])
        self.relu = nn.ELU()
        nn.init.kaiming_uniform_(self.conv0.weight)
    def forward(self,input):
        out = self.relu(self.conv0(input))
        return out
    

class fc_block(nn.Module):
    def __init__(self,start,init_shape):
        super(fc_block,self).__init__()
        shape = init_shape['shape']
        self.fc = nn.Linear(in_features=start,out_features=shape[0]*shape[1] + 2*shape[1])
        self.relu = nn.ELU()
        self.init_weight_network(self.fc.weight,shape[1])
    def forward(self,input):
        out = self.relu(self.fc(input))
        return out

    def init_weight_network(self,w,fan_out_dj):
        with torch.no_grad():
            fan_in = w.shape[1]
            a = torch.sqrt(torch.tensor(3*2/(fan_in*fan_out_dj)))
            return w.uniform_(-a,a)



class multiloss(nn.Module):
    def __init__(self,objective_num):
        super(multiloss,self).__init__()
        self.objective_num = objective_num
        self.log_var = nn.Parameter(torch.zeros(self.objective_num))
    
    def forward(self,losses):
        for i in range(len(losses)):
            precision = torch.exp(-self.log_var[i])
            if i == 0:
                loss = precision*losses[i] + self.log_var[i]
            else :
                loss += precision*losses[i] + self.log_var[i]
        return loss


class fine_tuner_hyper(nn.Module):
    def __init__(self,fc_parameters):
        super(fine_tuner_hyper,self).__init__()
        cnt = 0
        for fc in fc_parameters:
            shape = fc['shape']
            cnt += shape[0]*shape[1] + 2*shape[1]
        self.metal_plane_feature_map = nn.Sequential(nn.Conv2d(16,1,3),nn.ELU(),nn.Conv2d(1,1,3),nn.ELU(),nn.Conv2d(1,1,3))
        self.fc = nn.Linear(3364,cnt)
        self.fc2 = nn.Linear(2*cnt,cnt)
        self.activation = nn.ELU()
        nn.init.kaiming_uniform(self.fc.weight)
        self.init_weight_network(self.fc2.weight,32)
    def forward(self,weights,mp):
        mp = self.metal_plane_feature_map(mp).squeeze().view(1,-1)
        w = self.activation(self.fc2(torch.cat((weights,self.fc(self.activation(mp))),dim= 1)))
        return w

    def init_weight_network(self,w,fan_out_dj):
        with torch.no_grad():
            fan_in = w.shape[1]
            a = torch.sqrt(torch.tensor(3*2/(fan_in*fan_out_dj)))
            return w.uniform_(-a,a)



class fine_tuner(nn.Module):
    def __init__(self):
        super(fine_tuner,self).__init__()
        self.metal_plane_feature_map = nn.Sequential( resnet.ResNetBasicBlock(16,16), resnet.ResNetBasicBlock(16,16),
        resnet.ResNetBasicBlock(16,16))
        self.dec_cube_feature_map = nn.Sequential( resnet.ResNetBasicBlock(16,16), resnet.ResNetBasicBlock(16,16),
        resnet.ResNetBasicBlock(16,16))
        self.combining = resnet.ResNetBasicBlock(in_channels = 32,out_channels = 16)
        self.weights = nn.Parameter(torch.ones(2))
        self.activation = nn.Identity() 
    def forward(self,decision_cube,metal_plane):
        mp = self.metal_plane_feature_map(metal_plane)
        dc = self.dec_cube_feature_map(decision_cube)
        combined = self.combining(torch.cat((mp,dc),dim=1))
        #print(mp.shape)
        #print(di.shape)
        #print(combined.shape)
        output = self.activation(self.weights[0]*combined+ self.weights[1]*decision_cube)
        return output

class baseline_network(nn.Module):
    def __init__(self):
        super(baseline_network,self).__init__()
        self.directivity_conv = nn.Sequential( resnet.ResNetBasicBlock(1,16), resnet.ResNetBasicBlock(16,16),
        resnet.ResNetBasicBlock(16,16),resnet.ResNetBasicBlock(16,16),resnet.ResNetBasicBlock(16,16), resnet.ResNetBasicBlock(16,16))
        #self.weight_2 = nn.Parameter(torch.ones(64*64*16))
        #self.bias_2 = nn.Parameter(torch.zeros(64*64*16))
        #self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,directivity):
        dec = self.directivity_conv(directivity)
        #dec = self.sigmoid((dec.view(-1)*self.weight_2) + self.bias_2)
        output = self.sigmoid(dec.view(1,16,64,64))
        return output
import torch
import torch.nn as nn
import numpy as np
import cost
import resnet

class designer_network(nn.Module):
    def __init__(self,fc_shape):
        super(designer_network,self).__init__()
        self.hypernet = hypernetwork_constarints(fc_parameters=fc_shape)
        self.relu = nn.ELU()
        self.activation = nn.Sigmoid()
        self.fc_shape = fc_shape
    def forward(self,input):
        cord,constraints = input
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
    def forward(self,input):
        (cord,constraints),metal_plane = input
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
        
        self.relu = nn.ELU()
        self.directivity_conv = nn.Sequential( resnet.ResNetBasicBlock(1,16),nn.MaxPool2d(2), resnet.ResNetBasicBlock(16,16),
        resnet.ResNetBasicBlock(16,16), resnet.ResNetBasicBlock(16,1))
        self.fc = nn.Linear(1024+3,1024)
        cnt = 0
        for fc in fc_parameters:
            shape = fc['shape']
            cnt += shape[0]*shape[1] + 2*shape[1]
        
        self.fc2 = nn.Linear(1024,cnt)
        self.init_weight_network(self.fc2.weight,64)
        #nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc.weight)
        #self.metal_plane_conv = nn.Sequential(conv_block([16,32,3]),nn.MaxPool2d(2),conv_block([32,16,3]),conv_block([16,8,3]),nn.MaxPool2d(2),conv_block([8,1,3]))
        # size of metal plane = 121

        
    def forward(self,inputs):
        directivity,s = inputs
        directivity = self.directivity_conv(directivity)
        concat = directivity.view(1,-1)
        concat = torch.cat((concat,s),dim = 1)
        concat = self.relu(self.fc(concat))
        concat = self.relu(self.fc2(concat))
        return concat
 
    def init_bias_network(self,w):
        with torch.no_grad():
            fan_in = w.shape[0]
            uni = torch.randn(w.shape)
            a = torch.sqrt(torch.tensor(1/fan_in))
            w = uni*a
            return w    

    def init_weight_network(self,w,fan_out_dj):
        with torch.no_grad():
            fan_in = w.shape[0]
            a = torch.sqrt(torch.tensor(3/(fan_in*fan_out_dj)))
            return w.uniform_(-a,a)

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
            fan_in = w.shape[0]
            a = torch.sqrt(torch.tensor(3/(fan_in*fan_out_dj)))
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
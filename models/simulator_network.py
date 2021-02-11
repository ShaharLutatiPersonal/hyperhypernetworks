import torch.nn as nn
import torch
from torch.utils import data
import os
from scipy.io import loadmat
import PIL
from torch.utils.data.sampler import Sampler
from random import shuffle
import numpy as np
from PIL import Image
import resnet

def imresize(x,h,w):
    return np.array(Image.fromarray(x).resize(size=(h, w)))

factor = 113*113

def rel_snr(output,tar_img):
    eps = 1e-9
    tar_img = tar_img + 1e-12
    loss = torch.sum(torch.sum((output - tar_img).pow(2)))
    norm_factor = torch.sum(torch.sum(tar_img.pow(2))) 
    loss = (loss+eps)/(norm_factor+eps)
    return 10*torch.log10(loss/weight),loss,weight

def divide_to_minibatch(lst,idx,minibatch):
    tmp_lst = [lst[idxs] for idxs in idx]
    new_lst = []
    while len(tmp_lst) > minibatch:
        new_lst.append(tmp_lst[:minibatch])
        try:
            for i in range(minibatch):
                tmp_lst.pop(i)
        except Exception:
            continue
    new_lst.append(tmp_lst)
    return new_lst


class dataset(data.Dataset):
    def __init__(self,geometry_main_folder,data_main_folder,max_folder = 10000,mini_batch_size = 40):
        super(dataset,self).__init__()
        self.data_folder = data_main_folder
        self.max_folder = max_folder
        self.folder_list = [os.path.join(geometry_main_folder, o) for o in os.listdir(geometry_main_folder) 
                    if (os.path.isdir(os.path.join(geometry_main_folder,o)) and (int(o.split('_')[-1])<self.max_folder) and (os.path.isfile(os.path.join(geometry_main_folder, o) + '//geometry.pt')))]
        data_geom_tupple = [torch.load(folder + '//' + 'geometry.pt') for folder in self.folder_list]
        self.data_vec = []
        self.data_geom = []
        self.minibatch_size = mini_batch_size
        for t in data_geom_tupple:
            g,v = t
            self.data_geom.append(g)
            self.data_vec.append(v)
        self.data_res = [loadmat(self.data_folder + '//' + 'sim_no_{}.mat'.format(folder.split('_')[-1]))['rad'] for folder in self.folder_list]
        print(len(self.data_res))
        self.minibatch_orig = [(g,v,d) for g,v,d in zip(self.data_geom,self.data_vec,self.data_res)]
        self.train_indecis = [i for i in range(len(self.minibatch_orig))]
        self.cnt = np.ceil(len(self.minibatch_orig)/mini_batch_size)
        self.minibatch = divide_to_minibatch(self.minibatch_orig,self.train_indecis,mini_batch_size)
        print(len(self.minibatch))
        print('done loading data to cpu RAM')
    def __len__(self):
        return len(self.minibatch)
    def shuffle(self):
        shuffle(self.train_indecis)
        self.minibatch = divide_to_minibatch(self.minibatch_orig,self.train_indecis,self.minibatch_size)
    def __getitem__(self,index):
        if index == (len(self.minibatch) - 1):
            print("Warning this batch is for evaluation ! (not complete batch)")
        data_list_of_tupple = self.minibatch[index]
        for i,(g,v,d) in enumerate(data_list_of_tupple):
            if i == 0:
                geometry = (100*g.type(torch.float)).unsqueeze(0)
                result = 180 + 10*torch.log10(torch.tensor(imresize(d,64,64))).unsqueeze(0)
                coordinate = v.type(torch.float).unsqueeze(0)
            else:
                g_tmp = (100*g.type(torch.float)).unsqueeze(0)
                res_tmp = (180 + 10*torch.log10(torch.tensor(imresize(d,64,64)))).unsqueeze(0)
                cord_tmp = (v.type(torch.float)).unsqueeze(0)
                geometry = torch.cat([geometry,g_tmp],dim=0)
                result =  torch.cat((result,res_tmp),0)
                coordinate = torch.cat((coordinate,cord_tmp),0)
        return geometry,coordinate,result
    
    def fix_meas(self,input):
        shape = input.shape
        if shape[2] == 16:
            input = torch.cat([input,(255.0*torch.ones(512,512)).unsqueeze(2)],dim = 2)
        return 360*(1-(input/255.0))

class DataLoader(object):
    def __init__(self, dataset = dataset, batch_size = 1, drop_last=True):
        self.ds = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = [x for x in range(self.ds.__len__())]
        shuffle(self.sampler)
        self.current = 0

    def __iter__(self):
        for index in self.sampler:
            index = self.sampler[self.current]
            geom,res = self.ds[index]
            yield geom,res



class simulator_cnn(nn.Module):
    def __init__(self):
        super(simulator_cnn,self).__init__()
        kernel_size_first_conv = 3
        self.relu = nn.ReLU()
        self.hypernet = hypernet(kernel_size_first_conv,16,1)
        self.elu = nn.ELU()
        self.downscale = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,bias=True)
        self.net = nn.Sequential(nn.Dropout2d(0.3),resnet.ResNetBasicBlock(1,32),nn.Dropout2d(0.3),resnet.ResNetBasicBlock(32,1),nn.Dropout2d(0.3),resnet.ResNetBasicBlock(32,1))
        self.linear = nn.Linear(in_features= 60*60,out_features= 64*64,bias=True)
        self.deep_cnn_2 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size= 1,bias=True)
        torch.nn.init.kaiming_uniform_(self.downscale.weight)
        torch.nn.init.kaiming_uniform_(self.linear.weight)
    def forward(self,inp):
        input_s,vector = inp
        batch_size = vector.shape[0]
        input_s = input_s.permute(0,3,1,2).contiguous()
        for i in range(batch_size):
            tmp_vector = vector[i,:,:]
            weight = self.hypernet(tmp_vector.view(1,-1))
            tmp_input = (input_s[i,:,:,:]).unsqueeze(0)
            output = self.elu(torch.nn.functional.conv2d(tmp_input,weight))
            if i == 0:
                output_batchified = output
            else:
                output_batchified = torch.cat((output_batchified,output),dim=0)
        output = self.elu(self.downscale(output_batchified))
        output = (self.elu(self.net(output)))
        output = self.elu(self.linear(output.view(batch_size,-1)).view(batch_size,1,64,64))
        output = self.elu(self.deep_cnn_2(output))
        return output


class hypernet(nn.Module):
    def __init__(self,kernel_size,input_channels,output_channels):
        super(hypernet,self).__init__()
        self.elu = nn.ELU()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        total_size = kernel_size*kernel_size*input_channels*output_channels
        self.linear_1 = nn.Linear(in_features=6,out_features= total_size,bias=True)
        self.linear_3 = nn.Linear(in_features= total_size,out_features= total_size,bias=True)
        torch.nn.init.kaiming_uniform_(self.linear_1.weight)
        self.init_weight_network(self.linear_3.weight,3)
    def forward(self,input):
        output = self.elu(self.linear_1(input))
        output = self.elu(self.linear_3(output)).view(self.output_channels,self.input_channels,self.kernel_size,self.kernel_size)
        return output
    def init_weight_network(self,w,fan_out_dj):
        with torch.no_grad():
            fan_in = w.shape[1]
            a = torch.sqrt(torch.tensor(3*2/(fan_in*fan_out_dj)))
            return w.uniform_(-a,a)

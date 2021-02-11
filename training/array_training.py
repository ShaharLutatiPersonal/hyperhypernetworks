import torch
import torch.nn as nn
import array_network_resnet  
import numpy as np
from scipy.io import loadmat
import ctypes
from random import shuffle
import matplotlib.pyplot as plt
import pytorch_msssim
from PIL import Image
import simulator_network
import matplotlib.pyplot as plt
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
torch.cuda.empty_cache()
device = torch.device('cuda')



# Some initial computation 
ssim =  pytorch_msssim.ms_ssim
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

array_factor = torch.tensor(factor).to('cuda')


################## Utils for handeling database and losses ##########################

import utils
def show_image(x):
    fig,ax = plt.subplots()
    ax.imshow(x)
    fig.show()

def imresize(x,h,w):
    return np.array(Image.fromarray(x).resize(size=(h, w)))

import torchvision.transforms.functional as TF
from torchvision import transforms


def extract_single_antenna(G,L,cnt_len,i):
    if i == 0:
        prev = 0
    else:
        prev = sum(cnt_len[:i])
    return G[prev:prev + cnt_len[i],],L[prev:prev + cnt_len[i],]

assigment_pos = [[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]]

def gen_metal_plane(pos):
    x = torch.zeros(1,64*3,64*2)
    for i in pos:
        x[0,assigment_pos[i][0]*64:assigment_pos[i][0]*64 + 64,assigment_pos[i][1]*64:assigment_pos[i][1]*64 + 64] = 1
    return x

def fix_L(label,coord,ranges,random_block_metal,index):
    co_x = coord[:,0].cpu().numpy() / 1000.0
    co_y = coord[:,1].cpu().numpy() / 1000.0
    ranges = ranges.cpu().numpy()
    mp = random_block_metal[0,index*64:(index+1)*64,index*64:(index+1)*64].cpu().numpy()
    for i in range(len(label)):
        x_ix = int(np.round((co_x[i] - ranges[0][0][0]) / ((ranges[0][0][1]-ranges[0][0][0])/63)))
        y_ix = int(np.round((co_y[i] - ranges[0][1][0]) / ((ranges[0][1][1]-ranges[0][1][0])/63)))
        if mp[x_ix,y_ix] == 1:
            label[i] = 0
    return label



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


def gen_voxel_from_points_tensor(cord,points,labels):
    voxel = torch.zeros(16,64,64).to(device)
    cord = cord.cpu().numpy()
    points = points.cpu().numpy()
    res = labels
    z_arr = points[:,2]
    x_arr = points[:,0]
    y_arr = points[:,1]
    z_arr =np.clip((np.round((z_arr - cord[0][2][0])/((cord[0][2][1]-cord[0][2][0])/15))),0,15)
    y_arr = np.clip((np.round((y_arr - cord[0][1][0])/((cord[0][1][1]-cord[0][1][0])/63))),0,63)
    x_arr = np.clip((np.round((x_arr - cord[0][0][0])/((cord[0][0][1]-cord[0][0][0])/63))),0,63)
    voxel[z_arr,x_arr,y_arr] = res
    return voxel

def simulator_wrapper(labels,coord,ranges,cnt_len):
    for i in range(6):
        G,o = extract_single_antenna(coord,labels,cnt_len,i)
        fixed_range = fix_ranges(ranges[i],G/1000.0)
        voxel = (gen_voxel_from_points_tensor(fixed_range*1000,G,o)).to('cuda')
        rr = fixed_range.to('cuda')
        sim_out_res = sim_net(((voxel*120).squeeze().permute(2,1,0).clamp_max_(100).unsqueeze(0),rr)).squeeze()
        #sim_out_res = sim_out_res.to('cuda')
        if i == 0:
            R = sim_out_res.unsqueeze(0)
        else:
            R = torch.cat((R,sim_out_res.unsqueeze(0)),dim = 0)         
    return (utils.array_gain_calc_torch(R.to('cuda'),array_factor)).unsqueeze(0).unsqueeze(0)

def fix_ranges(r,g):
    x_max = torch.max(g[:,0])
    y_max = torch.max(g[:,1])
    z_max = torch.max(g[:,2])
    x_min = torch.min(g[:,0])
    y_min = torch.min(g[:,1])
    z_min = torch.min(g[:,2])
    r[0][0][1] = torch.max(torch.tensor([r[0][0][1],x_max]))
    r[0][1][1] = torch.max(torch.tensor([r[0][1][1],y_max]))
    r[0][2][1] = torch.max(torch.tensor([r[0][2][1],z_max]))
    r[0][0][0] = torch.min(torch.tensor([r[0][0][1],x_min]))
    r[0][1][0] = torch.min(torch.tensor([r[0][1][1],y_min]))
    r[0][2][0] = torch.min(torch.tensor([r[0][2][1],z_min]))
    return r

def constraint_loss(labels,coord,ranges_tmp,avoid_metal,cnt_len):
    res = 0
    for i in range(6):
        ranges = ranges_tmp[i].cpu().numpy()
        if i == 0:
            prev = 0
        else:
            prev = sum(cnt_len[:i])
        L = labels[prev:prev + cnt_len[i],]
        cordi = coord[prev:prev + cnt_len[i],]
        ranges = fix_ranges(ranges,cordi/1000.0)
        co_x = cordi[:,0].cpu().numpy() / 1000.0
        co_y = cordi[:,1].cpu().numpy() / 1000.0
        mp = avoid_metal[0,i*64:(i+1)*64,i*64:(i+1)*64].cpu().numpy()
        ix = np.round((co_x - ranges[0][0][0]) / ((ranges[0][0][1]-ranges[0][0][0])/63))
        iy = np.round((co_y - ranges[0][1][0]) / ((ranges[0][1][1]-ranges[0][1][0])/63))
        idx = torch.tensor([1 if mp[int(xx),int(yy)] else 0 for xx,yy in zip(ix,iy)]).to(device)
        res += torch.mean(clap_log(L)*idx)
    return res

def clapped_log(x):
    return torch.clip(10*torch.log10(x+1e-10),-100,10)


import torchvision.transforms.functional as TF
from torchvision import transforms

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = []
        self.totensor = transforms.ToTensor()

    def __call__(self, x,angle):
        x = transforms.ToPILImage()(x)
        x.rotate(angle)
        return self.totensor(x)

################## Training sequence ##########################
model = array_network_resnet.resnet_hyperhyper_variant().to(device).float()
lr = float(input('Learning rate?\n'))
l1 = nn.L1Loss()
clip_norm = 1
num_epochs = int(input('How many epochs?\n'))
crossentropyloss = nn.CrossEntropyLoss(weight=torch.tensor([1,6]).float().to(device))
decay_factor = float(input('Decay factor?\n'))
l = 0
p = torch.tensor(np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[2,0,0],[2,1,0]])*1000*0.45*3e8/2.4e9)
def half_bce(source,target):
    loss = -(source)*clapped_log(1-target)
    loss = torch.mean(loss)
    return loss


obceloss = half_bce
sigmoid = torch.sigmoid
multiloss_func = multiloss(2).to(device)
optim = torch.optim.Adam(model.parameters(),lr)
inputs_path = ''
Awesome_list =   torch.load(inputs_path) 
n_examples = int(len(Awesome_list)*3/4)
grad_accum = int(input('How many examples in minibatch?\n')) # equivalent to minibatch in term of loss averaging.
import time
loss_array_logger = []
for epoch in range(num_epochs):
    cnt = 0
    cum_error = 0
    optim.zero_grad()
    cnt_avg_p = 0 
    cnt_avg_n = 0
    true_positives_avg = 0 
    false_positives_avg = 0 
    true_negatives_avg = 0 
    false_negatives_avg = 0 
    for nexp in range(n_examples):
        #t_s = time.time()
        g_input,G,L,random_block_metal,ranges_tmp,cnt_len,s = Awesome_list[nexp]
        g_input = g_input.to(device)
        G = G.to(device)
        L = L.to(device)
        random_block_metal = random_block_metal.to(device)   
        #t_inputs = time.time()-t_s
        output = model(g_input.to(device),G.float(),random_block_metal.unsqueeze(0).to(device),s.to(device)).squeeze()
        fake_antenna = torch.ones(output.shape[0])
        fake_antenna[sum(cnt_len):] = 0 
        celit = crossentropyloss(output[:,:2],L.long())
        logist_loss = obceloss(fake_antenna.to(device),sigmoid(output[:,2]))
        l =  multiloss_func([celit,logist_loss])
        loss_f = l
        loss_array_logger.append(loss_f.item())
        loss_f = loss_f/grad_accum
        loss_f.backward()
        torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_norm )
        if (nexp+1) % grad_accum == 0:
            optim.step()
            model.zero_grad()
        
        with torch.no_grad():
            out_dec = output[:,1] > output[:,0]
            label = L
            true_positives, false_positives, true_negatives, false_negatives = confusion(out_dec, label)
            cnt_avg_p += torch.sum(label).item()
            cnt_avg_n += label.shape[0] - torch.sum(label).item()
            true_positives_avg += true_positives
            false_positives_avg += false_positives
            true_negatives_avg += true_negatives
            false_negatives_avg += false_negatives

        cnt += 1
        if cnt % 100 == 0 :
            print('epoch {} , {}/{} , cum_error (so far) {} , tp {} fn {} tn {} fp {}'.format(epoch,cnt,n_examples,cum_error/cnt,
            true_positives_avg/cnt_avg_p,false_negatives_avg/cnt_avg_p,true_negatives_avg/cnt_avg_n,false_positives_avg/cnt_avg_n))
    optim.param_groups[0]['lr'] *= decay_factor


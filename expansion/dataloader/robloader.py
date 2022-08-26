import os
import numbers
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import torchvision
from . import flow_transforms
import pdb
import cv2
from utils.flowlib import read_flow
from utils.util_flow import readPFM


def default_loader(path):
    return Image.open(path).convert('RGB')

def flow_loader(path):
    if '.pfm' in path:
        data =  readPFM(path)[0]
        data[:,:,2] = 1
        return data
    else:
        return read_flow(path)
        

def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:    
        return readPFM(path)[0]

class myImageFloder(data.Dataset):
    def __init__(self, iml0, iml1, flowl0, loader=default_loader, dploader= flow_loader, scale=1.,shape=[320,448], order=1, noise=0.06, pca_augmentor=True, prob = 1., cover=False, black=False, scale_aug=[0.4,0.2]):
        self.iml0 = iml0
        self.iml1 = iml1
        self.flowl0 = flowl0
        self.loader = loader
        self.dploader = dploader
        self.scale=scale
        self.shape=shape
        self.order=order
        self.noise = noise
        self.pca_augmentor = pca_augmentor
        self.prob = prob
        self.cover = cover
        self.black = black
        self.scale_aug = scale_aug

    def __getitem__(self, index):
        iml0  = self.iml0[index]
        iml1 = self.iml1[index]
        flowl0= self.flowl0[index]
        th, tw = self.shape

        iml0 = self.loader(iml0)
        iml1 = self.loader(iml1)
        iml1 = np.asarray(iml1)/255.
        iml0 = np.asarray(iml0)/255.
        iml0 = iml0[:,:,::-1].copy()
        iml1 = iml1[:,:,::-1].copy()
        flowl0 = self.dploader(flowl0)
        #flowl0[:,:,-1][flowl0[:,:,0]==np.inf]=0  # for gtav window pfm files
        #flowl0[:,:,0][~flowl0[:,:,2].astype(bool)]=0  
        #flowl0[:,:,1][~flowl0[:,:,2].astype(bool)]=0  # avoid nan in grad
        flowl0 = np.ascontiguousarray(flowl0,dtype=np.float32)
        flowl0[np.isnan(flowl0)] = 1e6 # set to max

        ## following data augmentation procedure in PWCNet 
        ## https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/data_augmentation_layer.cu
        import __main__ # a workaround for "discount_coeff"
        try:
            with open('iter_counts-%d.txt'%int(__main__.args.logname.split('-')[-1]), 'r') as f:
                iter_counts = int(f.readline())
        except:
            iter_counts = 0
        schedule = [0.5, 1., 50000.]  # initial coeff, final_coeff, half life
        schedule_coeff = schedule[0] + (schedule[1] - schedule[0]) * \
          (2/(1+np.exp(-1.0986*iter_counts/schedule[2])) - 1)

        if self.pca_augmentor:
            pca_augmentor = flow_transforms.pseudoPCAAug( schedule_coeff=schedule_coeff)
        else:
            pca_augmentor = flow_transforms.Scale(1., order=0)

        if np.random.binomial(1,self.prob):
            co_transform = flow_transforms.Compose([
            flow_transforms.Scale(self.scale, order=self.order),
            #flow_transforms.SpatialAug([th,tw], trans=[0.2,0.03], order=self.order, black=self.black),
            flow_transforms.SpatialAug([th,tw],scale=[self.scale_aug[0],0.03,self.scale_aug[1]],
                                               rot=[0.4,0.03],
                                               trans=[0.4,0.03],
                                               squeeze=[0.3,0.], schedule_coeff=schedule_coeff, order=self.order, black=self.black),
            #flow_transforms.pseudoPCAAug(schedule_coeff=schedule_coeff),
            flow_transforms.PCAAug(schedule_coeff=schedule_coeff),
            flow_transforms.ChromaticAug( schedule_coeff=schedule_coeff, noise=self.noise),
            ])
        else:
            co_transform = flow_transforms.Compose([
            flow_transforms.Scale(self.scale, order=self.order),
            flow_transforms.SpatialAug([th,tw], trans=[0.4,0.03], order=self.order, black=self.black)
            ])

        augmented,flowl0 = co_transform([iml0, iml1], flowl0)
        iml0 = augmented[0]
        iml1 = augmented[1]

        if self.cover:
            ## randomly cover a region
            # following sec. 3.2 of http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Hierarchical_Deep_Stereo_Matching_on_High-Resolution_Images_CVPR_2019_paper.html
            if np.random.binomial(1,0.5):
                #sx = int(np.random.uniform(25,100))
                #sy = int(np.random.uniform(25,100))
                sx = int(np.random.uniform(50,125))
                sy = int(np.random.uniform(50,125))
                #sx = int(np.random.uniform(50,150))
                #sy = int(np.random.uniform(50,150))
                cx = int(np.random.uniform(sx,iml1.shape[0]-sx))
                cy = int(np.random.uniform(sy,iml1.shape[1]-sy))
                iml1[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(iml1,0),0)[np.newaxis,np.newaxis]

        iml0  = torch.Tensor(np.transpose(iml0,(2,0,1)))
        iml1  = torch.Tensor(np.transpose(iml1,(2,0,1)))

        return iml0, iml1, flowl0

    def __len__(self):
        return len(self.iml0)

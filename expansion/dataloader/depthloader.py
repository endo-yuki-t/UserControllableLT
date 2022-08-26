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
from . import depth_transforms as flow_transforms
import pdb
import cv2
from utils.flowlib import read_flow
from utils.util_flow import readPFM, load_calib_cam_to_cam

def default_loader(path):
    return Image.open(path).convert('RGB')

def flow_loader(path):
    if '.pfm' in path:
        data =  readPFM(path)[0]
        data[:,:,2] = 1
        return data
    else:
        return read_flow(path)

def load_exts(cam_file):
    with open(cam_file, 'r') as f:
        lines = f.readlines()

    l_exts = []
    r_exts = []
    for l in lines:
        if 'L ' in l:
            l_exts.append(np.asarray([float(i) for i in l[2:].strip().split(' ')]).reshape(4,4))
        if 'R ' in l:
            r_exts.append(np.asarray([float(i) for i in l[2:].strip().split(' ')]).reshape(4,4))
    return l_exts,r_exts        

def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:    
        return readPFM(path)[0]

# triangulation
def triangulation(disp, xcoord, ycoord, bl=1, fl = 450, cx = 479.5, cy = 269.5):
    depth = bl*fl / disp # 450px->15mm focal length
    X = (xcoord - cx) * depth / fl
    Y = (ycoord - cy) * depth / fl
    Z = depth
    P = np.concatenate((X[np.newaxis],Y[np.newaxis],Z[np.newaxis]),0).reshape(3,-1)
    P = np.concatenate((P,np.ones((1,P.shape[-1]))),0)
    return P

class myImageFloder(data.Dataset):
    def __init__(self, iml0, iml1, flowl0, loader=default_loader, dploader= flow_loader, scale=1.,shape=[320,448], order=1, noise=0.06, pca_augmentor=True, prob = 1.,sc=False,disp0=None,disp1=None,calib=None ):
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
        self.sc = sc
        self.disp0 = disp0
        self.disp1 = disp1
        self.calib = calib

    def __getitem__(self, index):
        iml0  = self.iml0[index]
        iml1 = self.iml1[index]
        flowl0= self.flowl0[index]
        th, tw = self.shape

        iml0 = self.loader(iml0)
        iml1 = self.loader(iml1)

        # get disparity
        if self.sc:
            flowl0 = self.dploader(flowl0)
            flowl0 = np.ascontiguousarray(flowl0,dtype=np.float32)
            flowl0[np.isnan(flowl0)] = 1e6 # set to max
            if 'camera_data.txt' in self.calib[index]:
                bl=1
                if '15mm_' in self.calib[index]: 
                    fl=450 # 450
                else:
                    fl=1050
                cx = 479.5
                cy = 269.5
                # negative disp
                d1 = np.abs(disparity_loader(self.disp0[index]))
                d2 = np.abs(disparity_loader(self.disp1[index]) + d1)
            elif 'Sintel' in self.calib[index]:
                fl = 1000
                bl = 1
                cx = 511.5
                cy = 217.5
                d1 = np.zeros(flowl0.shape[:2])
                d2 = np.zeros(flowl0.shape[:2])
            else:
                ints = load_calib_cam_to_cam(self.calib[index])
                fl = ints['K_cam2'][0,0]
                cx = ints['K_cam2'][0,2]
                cy = ints['K_cam2'][1,2]
                bl = ints['b20']-ints['b30']
                d1 = disparity_loader(self.disp0[index])
                d2 = disparity_loader(self.disp1[index])
            #flowl0[:,:,2] = (flowl0[:,:,2]==1).astype(float)
            flowl0[:,:,2] = np.logical_and(np.logical_and(flowl0[:,:,2]==1, d1!=0), d2!=0).astype(float)

            shape = d1.shape
            mesh = np.meshgrid(range(shape[1]),range(shape[0]))
            xcoord = mesh[0].astype(float)
            ycoord = mesh[1].astype(float)
            
            # triangulation in two frames
            P0 = triangulation(d1, xcoord, ycoord, bl=bl, fl = fl, cx = cx, cy = cy)
            P1 = triangulation(d2, xcoord + flowl0[:,:,0], ycoord + flowl0[:,:,1], bl=bl, fl = fl, cx = cx, cy = cy)
            dis0 = P0[2]
            dis1 = P1[2]

            change_size =  dis0.reshape(shape).astype(np.float32)
            flow3d = (P1-P0)[:3].reshape((3,)+shape).transpose((1,2,0))

            gt_normal = np.concatenate((d1[:,:,np.newaxis],d2[:,:,np.newaxis],d2[:,:,np.newaxis]),-1)
            change_size = np.concatenate((change_size[:,:,np.newaxis],gt_normal,flow3d),2)
        else:
            shape = iml0.size
            shape=[shape[1],shape[0]]
            flowl0 = np.zeros((shape[0],shape[1],3))
            change_size = np.zeros((shape[0],shape[1],7))
            depth = disparity_loader(self.iml1[index].replace('camera','groundtruth'))
            change_size[:,:,0] = depth

            seqid = self.iml0[index].split('/')[-5].rsplit('_',3)[0]
            ints = load_calib_cam_to_cam('/data/gengshay/KITTI/%s/calib_cam_to_cam.txt'%seqid)
            fl = ints['K_cam2'][0,0]
            cx = ints['K_cam2'][0,2]
            cy = ints['K_cam2'][1,2]
            bl = ints['b20']-ints['b30']


        iml1 = np.asarray(iml1)/255.
        iml0 = np.asarray(iml0)/255.
        iml0 = iml0[:,:,::-1].copy()
        iml1 = iml1[:,:,::-1].copy()

        ## following data augmentation procedure in PWCNet 
        ## https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/data_augmentation_layer.cu
        import __main__ # a workaround for "discount_coeff"
        try:
            with open('/scratch/gengshay/iter_counts-%d.txt'%int(__main__.args.logname.split('-')[-1]), 'r') as f:
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
            co_transform1 = flow_transforms.Compose([
                           flow_transforms.SpatialAug([th,tw],
                                           scale=[0.2,0.,0.1],
                                           rot=[0.4,0.],
                                           trans=[0.4,0.],
                                           squeeze=[0.3,0.], schedule_coeff=schedule_coeff, order=self.order),
            ])
        else:
            co_transform1 = flow_transforms.Compose([
            flow_transforms.RandomCrop([th,tw]),
            ])

        co_transform2 = flow_transforms.Compose([
            flow_transforms.pseudoPCAAug( schedule_coeff=schedule_coeff),
            #flow_transforms.PCAAug(schedule_coeff=schedule_coeff),
            flow_transforms.ChromaticAug( schedule_coeff=schedule_coeff, noise=self.noise),
            ])

        flowl0 = np.concatenate([flowl0,change_size],-1)
        augmented,flowl0,intr = co_transform1([iml0, iml1], flowl0, [fl,cx,cy,bl])
        imol0 = augmented[0]
        imol1 = augmented[1]
        augmented,flowl0,intr = co_transform2(augmented, flowl0, intr)

        iml0 = augmented[0]
        iml1 = augmented[1]
        flowl0 = flowl0.astype(np.float32)
        change_size = flowl0[:,:,3:]
        flowl0 = flowl0[:,:,:3]

        # randomly cover a region
        sx=0;sy=0;cx=0;cy=0
        if np.random.binomial(1,0.5):
            sx = int(np.random.uniform(25,100))
            sy = int(np.random.uniform(25,100))
            #sx = int(np.random.uniform(50,150))
            #sy = int(np.random.uniform(50,150))
            cx = int(np.random.uniform(sx,iml1.shape[0]-sx))
            cy = int(np.random.uniform(sy,iml1.shape[1]-sy))
            iml1[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(iml1,0),0)[np.newaxis,np.newaxis]

        iml0  = torch.Tensor(np.transpose(iml0,(2,0,1)))
        iml1  = torch.Tensor(np.transpose(iml1,(2,0,1)))

        return iml0, iml1, flowl0, change_size, intr, imol0, imol1, np.asarray([cx-sx,cx+sx,cy-sy,cy+sy])

    def __len__(self):
        return len(self.iml0)

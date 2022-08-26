from __future__ import print_function
import sys
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
cudnn.benchmark = False

class Expansion():
    
    def __init__(self, loadmodel = 'pretrained_models/optical_expansion/robust.pth', testres = 1, maxdisp = 256, fac = 1):       
        
        maxw,maxh = [int(testres*1280), int(testres*384)]
        
        max_h = int(maxh // 64 * 64)
        max_w = int(maxw // 64 * 64)
        if max_h < maxh: max_h += 64
        if max_w < maxw: max_w += 64
        maxh = max_h
        maxw = max_w
        
        mean_L = [[0.33,0.33,0.33]]
        mean_R = [[0.33,0.33,0.33]]
        
        # construct model, VCN-expansion
        from expansion.models.VCN_exp import VCN
        model = VCN([1, maxw, maxh], md=[int(4*(maxdisp/256)),4,4,4,4], fac=fac, 
          exp_unc=('robust' in loadmodel))  # expansion uncertainty only in the new model
        model = nn.DataParallel(model, device_ids=[0])
        model.cuda()
        
        if loadmodel is not None:
            pretrained_dict = torch.load(loadmodel)
            mean_L=pretrained_dict['mean_L']
            mean_R=pretrained_dict['mean_R']
            pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
            model.load_state_dict(pretrained_dict['state_dict'],strict=False)
        else:
            print('dry run')
        
        model.eval()
        # resize
        maxh = 256
        maxw = 256
        max_h = int(maxh // 64 * 64)
        max_w = int(maxw // 64 * 64)
        if max_h < maxh: max_h += 64
        if max_w < maxw: max_w += 64
        
        # modify module according to inputs
        from expansion.models.VCN_exp import WarpModule, flow_reg
        for i in range(len(model.module.reg_modules)):
            model.module.reg_modules[i] = flow_reg([1,max_w//(2**(6-i)), max_h//(2**(6-i))], 
                            ent=getattr(model.module, 'flow_reg%d'%2**(6-i)).ent,\
                            maxdisp=getattr(model.module, 'flow_reg%d'%2**(6-i)).md,\
                            fac=getattr(model.module, 'flow_reg%d'%2**(6-i)).fac).cuda()
        for i in range(len(model.module.warp_modules)):
            model.module.warp_modules[i] = WarpModule([1,max_w//(2**(6-i)), max_h//(2**(6-i))]).cuda()
            
        mean_L = torch.from_numpy(np.asarray(mean_L).astype(np.float32).mean(0)[np.newaxis,:,np.newaxis,np.newaxis]).cuda()
        mean_R = torch.from_numpy(np.asarray(mean_R).astype(np.float32).mean(0)[np.newaxis,:,np.newaxis,np.newaxis]).cuda()
        
        self.max_h = max_h
        self.max_w = max_w
        self.model = model
        self.mean_L = mean_L
        self.mean_R = mean_R
        
    def run(self, imgL_o, imgR_o):
        model = self.model
        mean_L = self.mean_L
        mean_R = self.mean_R
        
        imgL_o[imgL_o<-1] = -1
        imgL_o[imgL_o>1] = 1
        imgR_o[imgR_o<-1] = -1
        imgR_o[imgR_o>1] = 1
        imgL = (imgL_o+1.)*0.5-mean_L
        imgR = (imgR_o*1.)*0.5-mean_R
        
        with torch.no_grad():
            imgLR = torch.cat([imgL,imgR],0)
            model.eval()
            torch.cuda.synchronize()
            rts = model(imgLR)
            torch.cuda.synchronize()
            flow, occ, logmid, logexp = rts
        
        torch.cuda.empty_cache()
        
        return flow, logexp

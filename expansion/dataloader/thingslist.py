import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
  exc_list = [
'0004117.flo',
'0003149.flo',
'0001203.flo',
'0003147.flo',
'0003666.flo',
'0006337.flo',
'0006336.flo',
'0007126.flo',
'0004118.flo',
]

  left_fold  = 'image_clean/left/'
  flow_noc   = 'flow/left/into_future/'
  train = [img for img in os.listdir(filepath+flow_noc) if np.sum([(k in img) for k in exc_list])==0]

  l0_trainlf  = [filepath+left_fold+img.replace('flo','png') for img in train]
  l1_trainlf = ['%s/%s.png'%(img.rsplit('/',1)[0],'%07d'%(1+int(img.split('.')[0].split('/')[-1])) ) for img in l0_trainlf]
  flow_trainlf = [filepath+flow_noc+img for img in train]


  exc_list = [
'0003148.flo',
'0004117.flo',
'0002890.flo',
'0003149.flo',
'0001203.flo',
'0003666.flo',
'0006337.flo',
'0006336.flo',
'0004118.flo',
]

  left_fold  = 'image_clean/right/'
  flow_noc   = 'flow/right/into_future/'
  train = [img for img in os.listdir(filepath+flow_noc) if np.sum([(k in img) for k in exc_list])==0]

  l0_trainrf = [filepath+left_fold+img.replace('flo','png') for img in train]
  l1_trainrf = ['%s/%s.png'%(img.rsplit('/',1)[0],'%07d'%(1+int(img.split('.')[0].split('/')[-1])) ) for img in l0_trainrf]
  flow_trainrf = [filepath+flow_noc+img for img in train]


  exc_list = [
'0004237.flo',
'0004705.flo',
'0004045.flo',
'0004346.flo',
'0000161.flo',
'0000931.flo',
'0000121.flo',
'0010822.flo',
'0004117.flo',
'0006023.flo',
'0005034.flo',
'0005054.flo',
'0000162.flo',
'0000053.flo',
'0005055.flo',
'0003147.flo',
'0004876.flo',
'0000163.flo',
'0006878.flo',
]

  left_fold  = 'image_clean/left/'
  flow_noc   = 'flow/left/into_past/'
  train = [img for img in os.listdir(filepath+flow_noc) if np.sum([(k in img) for k in exc_list])==0]

  l0_trainlp  = [filepath+left_fold+img.replace('flo','png') for img in train]
  l1_trainlp = ['%s/%s.png'%(img.rsplit('/',1)[0],'%07d'%(-1+int(img.split('.')[0].split('/')[-1])) ) for img in l0_trainlp]
  flow_trainlp = [filepath+flow_noc+img for img in train]

  exc_list = [
'0003148.flo',
'0004705.flo',
'0000161.flo',
'0000121.flo',
'0004117.flo',
'0000160.flo',
'0005034.flo',
'0005054.flo',
'0000162.flo',
'0000053.flo',
'0005055.flo',
'0003147.flo',
'0001549.flo',
'0000163.flo',
'0006336.flo',
'0001648.flo',
'0006878.flo',
]

  left_fold  = 'image_clean/right/'
  flow_noc   = 'flow/right/into_past/'
  train = [img for img in os.listdir(filepath+flow_noc) if np.sum([(k in img) for k in exc_list])==0]

  l0_trainrp  = [filepath+left_fold+img.replace('flo','png') for img in train]
  l1_trainrp = ['%s/%s.png'%(img.rsplit('/',1)[0],'%07d'%(-1+int(img.split('.')[0].split('/')[-1])) ) for img in l0_trainrp]
  flow_trainrp = [filepath+flow_noc+img for img in train]


  l0_train = l0_trainlf + l0_trainrf + l0_trainlp + l0_trainrp
  l1_train = l1_trainlf + l1_trainrf + l1_trainlp + l1_trainrp
  flow_train = flow_trainlf + flow_trainrf + flow_trainlp + flow_trainrp
  return l0_train, l1_train, flow_train

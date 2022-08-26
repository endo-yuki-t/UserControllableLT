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

  left_fold  = 'image_2/'
  flow_noc   = 'flow_occ/'

  train = [img for img in os.listdir(filepath+left_fold) if 'Kitti' in img and img.find('_10') > -1]

#  train = [i for i in train if int(i.split('_')[1])%5==0]
  import pdb; pdb.set_trace()
  train = sorted([i for i in train if int(i.split('_')[1])%5==0])[0:1]

  l0_train  = [filepath+left_fold+img for img in train]
  l1_train = [filepath+left_fold+img.replace('_10','_11') for img in train]
  flow_train = [filepath+flow_noc+img for img in train]

  l0_train += [filepath+left_fold+img.replace('_10','_09') for img in train]
  l1_train += [filepath+left_fold+img for img in train]
  flow_train += flow_train

  tmp = l0_train
  l0_train = l0_train+ [i.replace('rob_flow', 'kitti_scene').replace('Kitti2015_','') for i in l1_train]
  l1_train = l1_train+tmp
  flow_train += flow_train

  return l0_train, l1_train, flow_train

import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  left_fold  = 'image_2/'
  train = [img for img in os.listdir(filepath+left_fold) if img.find('Sintel') > -1]

  l0_train  = [filepath+left_fold+img for img in train]
  l0_train  = [img for img in l0_train if '%s_%s.png'%(img.rsplit('_',1)[0],'%02d'%(1+int(img.split('.')[0].split('_')[-1])) ) in l0_train ]

  #l0_train = [i for i in l0_train if not '10.png' in i] # remove 10 as val

  l1_train = ['%s_%s.png'%(img.rsplit('_',1)[0],'%02d'%(1+int(img.split('.')[0].split('_')[-1])) ) for img in l0_train]
  flow_train = [img.replace('image_2','flow_occ') for img in l0_train]


  return l0_train, l1_train, flow_train

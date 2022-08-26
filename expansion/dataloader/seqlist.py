import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  train = [img for img in sorted(glob.glob('%s/*'%filepath))]

  l0_train  = train[:-1]
  l1_train = train[1:]


  return sorted(l0_train), sorted(l1_train), sorted(l0_train)

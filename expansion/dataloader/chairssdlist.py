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
    l0_train = []
    l1_train = []
    flow_train = []
    for flow_map in sorted(glob.glob('%s/flow/*.pfm'%filepath)):
        img1 = flow_map.replace('flow','t0').replace('.pfm','.png')
        img2 = flow_map.replace('flow','t1').replace('.pfm','.png')

        l0_train.append(img1)
        l1_train.append(img2)
        flow_train.append(flow_map)

    return l0_train, l1_train, flow_train

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
    for flow_map in sorted(glob.glob(os.path.join(filepath,'*_flow.flo'))):
        root_filename = flow_map[:-9]
        img1 = root_filename+'_img1.ppm'
        img2 = root_filename+'_img2.ppm'
        if not (os.path.isfile(os.path.join(filepath,img1)) and os.path.isfile(os.path.join(filepath,img2))):
            continue

        l0_train.append(img1)
        l1_train.append(img2)
        flow_train.append(flow_map)

    return l0_train, l1_train, flow_train

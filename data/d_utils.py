""" """
import logging
import sys
import random
import math
# Standard dist imports

# Third party imports
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms

# Project level imports
from utils.config import opt
from utils.constants import Constants as CONST

def pil_loader(path):
    return Image.open(path)

def data_transform(mode, img, label):
    ### Padding to Square
    img_size = img.size
    img_padding = [0,0]
    if img_size[0] > img_size[1]:
        img_padding[1] = int((img_size[0]-img_size[1])/2)
    else:
        img_padding[0] = int((img_size[1]-img_size[0])/2)
    img_padding = tuple(img_padding)
    [x,y] = [label[i]*img_size[i]+img_padding[i] for i in range(2)]
    img = transforms.Pad(img_padding,fill=0, padding_mode='constant')(img)
    label_ori_copy = [x/img.size[0], y/img.size[1]]
    
    #transform paramaters
    img_size = img.size
    angle_r = random.uniform(-180, 180)
    t_s = (0.2, 0.2)
    translate_r = [random.uniform(-1,1)*t_s[i]*img_size[i] for i in range(2)]
    horizontal_flip = random.randint(0,1)
    vertical_flip = random.randint(0,1)
    
    
    #transform flag
    valid = True
    
    if mode == 'train':
        # flips
        if horizontal_flip:
            x = img_size[0] - x
        if vertical_flip:
            y = img_size[0] - y
        # rotations
        if opt.rotation:
            theta = math.radians(angle_r)
            c = [img_size[0]/2, img_size[1]/2]
            cos_ = math.cos(theta)
            sin_ = math.sin(theta)
            label[0] = (x-c[0])*cos_ - (y-c[1])*sin_ + c[0]
            label[1] = (x-c[0])*sin_ + (y-c[1])*cos_ + c[1]
        else:
            label[0] = x
            label[1] = y
        label[0] = (label[0] + translate_r[0])/img_size[0]
        label[1] = (label[1] + translate_r[1])/img_size[1]
        #if phone is transformed out of the picture, then do not transform
        if sum([1 for x in label if (x<=0 or x >= 1)]):
            valid = False

    #image transformation
    if mode == CONST.TRAIN and valid:
        if horizontal_flip:
            img = transforms.functional.hflip(img)
        if vertical_flip:
            img = transforms.functional.vflip(img)
        if opt.rotation:
            img = transforms.functional.affine(img=img, angle=angle_r, 
                                               translate=translate_r, scale=1, shear=0,
                                               resample=Image.BICUBIC, fillcolor=0)
        else:
            img = transforms.functional.affine(img=img, angle=0, 
                                               translate=translate_r, scale=1, shear=0, fillcolor=0)
    else:
        label = label_ori_copy
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406]
                                                         ,std=[0.229, 0.224, 0.225])
                                    ])
    img = transform(img)
    
    return img, label, img_padding

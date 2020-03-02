"""Dataloader

#TODO DataLoader description needed

"""

# Standard dist imports
import os
import logging
import random

# Third party imports
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

# Project level imports
from data.d_utils import pil_loader, data_transform
from utils.config import opt
from utils.constants import Constants as CONST
from utils.logger import Logger


class PHONEDataset(Dataset):

    def __init__(self, data_dir=None, mode='train'):
        """Initializes PhoneDataset

        Args:
            data_dir (str): path to the image dir
            mode (str): Mode/partition of the dataset
        """
        assert mode in (CONST.TRAIN, CONST.VAL, CONST.DEPLOY), 'mode: train, val, deploy'
        self.mode = mode
        Logger.section_break(f'{self.mode.upper()} Dataset')
        self.logger = logging.getLogger('dataloader_'+mode)
        self.logger.setLevel(opt.logging_level)

        # === Read in the dataset ===#
        txt_fn = os.path.join(data_dir,'labels_{}.txt'.format(self.mode))
        if not os.path.exists(txt_fn):
            self.logger.info("Creating Train and Validation Dataset out of labels.txt")
            with open(os.path.join(data_dir, 'labels.txt')) as f:
                content = f.readlines()
            val_list = []
            val_n = max(1,int(len(content)*0.1))
            for i in range(val_n):
                random_ele = random.choice(content)
                val_list.append(random_ele)
                content.remove(random_ele)
            with open(os.path.join(data_dir,'labels_val.txt'),'w') as f:
                f.write(''.join(val_list))
            with open(os.path.join(data_dir,'labels_train.txt'),'w') as f:
                f.write(''.join(content))

        with open(txt_fn) as f:
            self.data = f.readlines()
        self.data = [x.strip().split(' ')[:3] for x in self.data]
        self.logger.info('Number of total images: {}'.format(len(self.data)))
        self.data = pd.DataFrame(self.data, columns = ['images', 'label_x', 'label_y'])
        for index, row in self.data.iterrows():
            row['images'] = os.path.join(data_dir, row['images'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            image and label tensor

        """
        # Load image
        img = pil_loader(self.data.iloc[index]['images'])

        # Load Label
        label = [self.data.iloc[index]['label_x'], self.data.iloc[index]['label_y']]
        label = [ float(x) for x in label]
        
        #Add img random transformation and adjust label accordingly
        img, label, _ = data_transform(mode = self.mode, img = img, label = label)
        label = torch.FloatTensor(label)
        return {'images': img, 'label': label}

def get_dataloader(data_dir=None, batch_size=1, shuffle=True,
                   num_workers=4, mode=CONST.TRAIN):
    """ Get the dataloader

    Args:
        mode (str):
        data_dir (str): Relative path to the csv data files
        csv_file (str): Absolute path of the csv file
        batch_size (int): Batch size
        shuffle (bool): Flag for shuffling dataset
        num_workers (int): Number of workers

    Returns:
        dict: Dictionary holding each type of dataloader

    """

    # Create a dataset from the training and validation csv files (consolidated in one data directory)
    logger = logging.getLogger('dataloader')
    dataset = PHONEDataset(data_dir=data_dir, mode=mode)

    # Create the dataloader from the given dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers,
                                               pin_memory=True)
    return data_loader

def to_cuda(item, computing_device, label=False):
    """ Typecast item to cuda()

    Wrapper function for typecasting variables to cuda() to allow for
    flexibility between different types of variables (i.e. long, float)

    Loss function usually expects LongTensor type for labels, which is why
    label is defined as a bool.

    Computing device is usually defined in the Trainer()

    Args:
        item: Desired item. No specific type
        computing_device (str): Desired computing device.
        label (bool): Flag to convert item to long() or float()

    Returns:
        item
    """
    if label:
        item = Variable(item.to(computing_device)).long()
    else:
        item = Variable(item.to(computing_device)).float()
    return item

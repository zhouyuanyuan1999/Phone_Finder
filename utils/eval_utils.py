from collections import OrderedDict
import itertools
import random
import logging
import math
import os

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import torch.nn as nn

from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

from utils.config import opt
from utils.constants import Constants as CONST

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.std = 0.

    def update(self, val, n=1):
        val = val.astype(float) if isinstance(val, np.ndarray) else float(val)
        self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = np.std(self.data)

def get_meter(meters=['batch_time', 'loss', 'acc']):
    return {meter_type: AverageMeter() for meter_type in meters}

def accuracy(predictions, targets, axis=1):
    batch_size = predictions.size(0)
    pred = np.float64(predictions.cpu().data.numpy())
    gtruth = np.float64(targets.cpu().data.numpy())
    dist = np.sum((pred - gtruth)**2,axis)**0.5
    hits = sum([1 for x in dist if x < 0.05])
    acc = 100. * float(hits) / float(batch_size)
    return acc

def vis_training(train_points, val_points, num_epochs=0, loss=True, **kwargs):
    """ Visualize losses and accuracies w.r.t each epoch

    Args:
        num_epochs: (int) Number of epochs
        train_points: (list) Points of the training curve
        val_points: (list) Points of the validation curve
        loss: (bool) Flag for loss or accuracy. Defaulted to True for loss

    """
    # Check if nan values in data points
    train_points = [i for i in train_points if not math.isnan(i)]
    val_points = [i for i in val_points if not math.isnan(i)]
    num_epochs = len(train_points)
    x = np.arange(0, num_epochs)

    plt.figure()
    plt.plot(x, train_points, 'b')
    plt.plot(x, val_points, 'r')

    title = '{} vs Number of Epochs'.format('Loss' if loss else 'Accuracy')
    if 'EXP' in kwargs:
        title += ' (EXP: {})'.format(kwargs['EXP'])
    plt.title(title)

    if loss:
        plt.ylabel('Cross Entropy Loss')
    else:
        plt.ylabel('Accuracy')

    plt.gca().legend(('Train', 'Val'))
    plt.xlabel('Number of Epochs')

    figs_folder_path = os.path.join(opt.model_dir, 'figs')
    if not os.path.exists(figs_folder_path):
        os.makedirs(figs_folder_path)
    
    save_path = os.path.join(opt.model_dir,'figs/train_val_{}'.format('loss' if loss else 'accuracy'))
    #save_path = './figs/train_val_{}'.format('loss' if loss else 'accuracy')
    for k_, v_ in kwargs.items():
        save_path += '_%s' % v_
    save_path += '.png'

    plt.savefig(save_path)
    plt.show()
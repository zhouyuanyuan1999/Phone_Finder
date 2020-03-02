"""Train and Evaluate Model

Script operates by requesting arguments from the user and feeds it into
`train_and_evaluate()`, which acts as the main() of the script.

Execution of the script is largely dependent upon the `--mode` of the model.
`train` will train the model and validate on a subset while `val` will go
through a full evaluation.

If the mode is set to `deploy`, then it will run the script assuming that
the model will be running on a test set (e.g. unseen/unlabelled data)

Logging is heavily incorporated into the script to track and log event
occurrences. If you are unfamiliar with `logging` please look into it.

"""
# Standard dist imports
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))


# Third party imports
import numpy as np
import pandas as pd
import torch

# Project level imports
from model.model import MODEL
from trainer import Trainer
from data.d_utils import pil_loader, data_transform
from data.dataloader import get_dataloader, to_cuda
from utils.constants import Constants as CONST
from utils.config import opt

#Predict only one image given the specified path
def deploy_single(img_path):
    # load model
    model = MODEL(arch=opt.arch, pretrained=False, num_classes=2)
    opt.resume = True
    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=CONST.VAL)

    # Load and Transform Image
    img = pil_loader(img_path)
    img_ori_size = np.array(img.size).copy()
    img, _, padding = data_transform(mode = CONST.VAL, img = img, label = [0,0])
    
    # Make Prediction
    model.eval()
    with torch.no_grad():
        img = to_cuda(img.unsqueeze(0), trainer.computing_device)
        pred = model(img)
        pred = np.float64(pred.cpu().data.numpy()).flatten()
    print(pred)
    pred = (pred*np.array(img.shape[2:]) - np.array(padding))/img_ori_size
    print("{} {}".format(pred[0],pred[1]))
    return 
 
#Prediction all jpg images in specified directory
def deploy_all(img_dir):
    # load model
    model = MODEL(arch=opt.arch, pretrained=False, num_classes=2)
    opt.resume = True
    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=CONST.VAL)
    
    for img_fn in os.listdir(img_dir):
        if img_fn[-3:] == 'jpg':
            img_path = os.path.join(img_dir,img_fn)
            # Load and Transform Image
            img = pil_loader(img_path)
            img_ori_size = np.array(img.size).copy()
            img, _, padding = data_transform(mode = CONST.VAL, img = img, label = [0,0])

            # Make Prediction
            model.eval()
            with torch.no_grad():
                img = to_cuda(img.unsqueeze(0), trainer.computing_device)
                pred = model(img)
                pred = np.float64(pred.cpu().data.numpy()).flatten()
            pred = (pred*np.array(img.shape[2:]) - np.array(padding))/img_ori_size
            print("{} {} {}".format(pred[0],pred[1],img_fn))
    return

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please add image path as an argument')
        sys.exit()
    path = sys.argv[1]
    if os.path.isfile(path):
        deploy_single(path)
    elif os.path.isdir(path):
        deploy_all(path)
    else:
        print('File or Directory not exist!')

    

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
import argparse
import logging
import os

import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from pprint import pformat
import time
from datetime import datetime

# Third party imports
import numpy as np
import torch
import matplotlib.pyplot as plt

# Project level imports
from model.model import MODEL
from trainer import Trainer
from data.dataloader import get_dataloader, to_cuda
from utils.constants import Constants as CONST
from utils.config import opt, set_config
from utils.eval_utils import accuracy, get_meter, vis_training
from utils.logger import Logger

# Module level constants

def train_and_evaluate(opt, logger=None):
    """ Train and evaluate a model

    The basic understanding of `train_and_evaluate()` can be broken down
    into two parts. Part 1 focuses on getting the dataloaders, model,
    and trainer to conduct the training/evaluation. Part 2.A and 2.B is about
    training or evaluating, respectively.

    Given the mode, train_and_evaluate can take two actions:

    1) mode == TRAIN ---> action: train_and_validate
    2) mode == VAL   ---> action: evaluate the model on the full validation/test set


    Args:
        opt (Config): A state dictionary holding preset parameters
        logger (Logger): Logging instance

    Returns:
        None

    """

    #TODO implement Early Stopping
    #TODO implement test code
    
    logger = logger if logger else logging.getLogger('train-and-evaluate')
    logger.setLevel(opt.logging_level)

    # Read in dataset
    # check the path for the data loader to make sure it is loading the right data set
    data_loader = {mode: get_dataloader(data_dir=opt.data_dir,
                                        batch_size=opt.batch_size,
                                        mode=mode) for mode in [CONST.TRAIN, CONST.VAL]}
    # Create model
    model = MODEL(arch=opt.arch, pretrained=opt.pretrained, num_classes=2)
    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=opt.mode,
                      resume=opt.resume, lr=opt.lr)

    #==== TRAINING ====#
    # Train and validate model if set to TRAINING
    # When training, we do both training and validation within the loop.
    # When set to the validation mode, this will run a full evaluation
    # and produce more summarized evaluation results. This is the default condition
    # if the mode is not training.
    if opt.mode == CONST.TRAIN:
        best_err = trainer.best_err
        Logger.section_break('Valid (Epoch {})'.format(trainer.start_epoch))
        err, acc, _ = evaluate(trainer.model, trainer, data_loader[CONST.VAL],
                                             0, opt.batch_size, logger)

        eps_meter = get_meter(meters=['train_loss', 'val_loss', 'train_acc', 'val_acc'])
        best_err = min(best_err, err)
        
        for ii, epoch in enumerate(range(trainer.start_epoch,
                                         trainer.start_epoch+opt.epochs)):

            # Train for one epoch
            Logger.section_break('Train (Epoch {})'.format(epoch))
            train_loss, train_acc = train(trainer.model, trainer, data_loader[CONST.TRAIN], 
                                          epoch, logger, opt.batch_size, opt.print_freq)
            eps_meter['train_loss'].update(train_loss)
            eps_meter['train_acc'].update(train_acc)
            
            # Evaluate on validation set
            Logger.section_break('Valid (Epoch {})'.format(epoch))
            err, acc, _ = evaluate(trainer.model, trainer, data_loader[CONST.VAL],
                                                 epoch, opt.batch_size, logger)
            eps_meter['val_loss'].update(err)
            eps_meter['val_acc'].update(acc)
                
            # Remember best error and save checkpoint
            is_best = err < best_err
            best_err = min(err, best_err)
            state = trainer.generate_state_dict(epoch=epoch, best_err=best_err)
            
            if is_best:
                trainer.save_checkpoint(state, is_best=is_best,
                                        filename='model_best.pth.tar')

        # ==== END: TRAINING LOOP ====#
        if len(eps_meter['train_loss'].data) > 0:
            #plot loss over eps
            vis_training(eps_meter['train_loss'].data, eps_meter['val_loss'].data, loss=True)
            #plot acc over eps
            vis_training(eps_meter['train_acc'].data, eps_meter['val_acc'].data, loss=False)

def train(model, trainer, train_loader, epoch, logger,
    batch_size=opt.batch_size, print_freq=opt.print_freq):
    """ Train the model

    Outside of the typical training loops, `train()` incorporates other
    useful bookkeeping features and wrapper functions. This includes things
    like keeping track of accuracy, loss, batch time to wrapping optimizers
    and loss functions in the `trainer`. Be sure to reference `trainer.py`
    or `utils/eval_utils.py` if extra detail is needed.

    Args:
        model: Classification model
        trainer (Trainer): Training wrapper
        train_loader (torch.data.Dataloader): Generator data loading instance
        epoch (int): Current epoch
        logger (Logger): Logger. Used to display/log metrics
        batch_size (int): Batch size
        print_freq (int): Print frequency

    Returns:
        None

    """
    criterion = trainer.criterion
    optimizer = trainer.optimizer

    # Initialize meter to bookkeep the following parameters
    meter = get_meter(meters=['batch_time', 'data_time', 'loss', 'acc'])

    # Switch to training mode
    model.train(True)

    end = time.time()
    for i, batch in enumerate(train_loader):
        # process batch items: images, labels
        img = to_cuda(batch[CONST.IMG], trainer.computing_device)
        target = to_cuda(batch[CONST.LBL], trainer.computing_device)

        # measure data loading time
        meter['data_time'].update(time.time() - end)

        # compute output
        end = time.time()
        end = time.time()
        logits = model(img)
        loss = criterion(logits, target)
        acc = accuracy(logits, target)        
        batch_size = list(batch[CONST.LBL].shape)[0]

        # update metrics
        meter['acc'].update(acc, batch_size)
        meter['loss'].update(loss, batch_size)
        
        # compute gradient and do sgd step
        optimizer.zero_grad()
        loss.backward()

        if i % print_freq == 0:
            log = 'TRAIN [{:02d}][{:2d}/{:2d}] TIME {:10} DATA {:10} ACC {:10} LOSS {:10}'.\
                format(epoch, i, len(train_loader),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['batch_time']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['data_time']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['acc']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['loss'])
                                )
            logger.info(log)

        optimizer.step()

        # measure elapsed time
        meter['batch_time'].update(time.time() - end)
        end = time.time()
    
    return meter['loss'].avg, meter['acc'].avg

    

def evaluate(model, trainer, data_loader, epoch=0,
             batch_size=opt.batch_size, logger=None):
    """ Evaluate model

    Similar to `train()` structure, where the function includes bookkeeping
    features and wrapper items. The only difference is that evaluation will
    only occur until the `max_iter` if it is specified and includes an
    `EvalMetrics` intiailization.

    The latter is currrently used to save predictions and ground truths to
    compute the confusion matrix.

    Args:
        model: Classification model
        trainer (Trainer): Training wrapper
        data_loader (torch.data.Dataloader): Generator data loading instance
        epoch (int): Current epoch
        logger (Logger): Logger. Used to display/log metrics
        batch_size (int): Batch size

    Returns:
        float: Loss average
        float: Accuracy average
        float: Run time average
        EvalMetrics: Evaluation wrapper to compute CMs

    """
    criterion = trainer.criterion

    # Initialize meter and metrics
    meter = get_meter(meters=['batch_time', 'loss', 'acc'])

    # Switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # process batch items: images, labels
            img = to_cuda(batch[CONST.IMG], trainer.computing_device)
            target = to_cuda(batch[CONST.LBL], trainer.computing_device)

            # compute output
            end = time.time()
            logits = model(img)
            loss = criterion(logits, target)
            acc = accuracy(logits, target)        
            batch_size = list(batch[CONST.LBL].shape)[0]

            # update metrics
            meter['acc'].update(acc, batch_size)
            meter['loss'].update(loss, batch_size)
            
            # measure elapsed time
            meter['batch_time'].update(time.time() - end, batch_size)

            if i % opt.print_freq == 0:
                log = 'EVAL [{:02d}][{:2d}/{:2d}] TIME {:10} ACC {:10} LOSS {' \
                      ':10}'.format(epoch, i, len(data_loader),
                    "{t.val:.3f} ({t.avg:.3f})".format(t=meter['batch_time']),
                    "{t.val:.3f} ({t.avg:.3f})".format(t=meter['acc']),
                    "{t.val:.3f} ({t.avg:.3f})".format(t=meter['loss'])
                                    )
                logger.info(log)

        # Print last eval
        log = 'EVAL [{:02d}][{:2d}/{:2d}] TIME {:10} ACC {:10} LOSS {' \
              ':10}'.format(epoch, i, len(data_loader),
                            "{t.val:.3f} ({t.avg:.3f})".format(t=meter['batch_time']),
                            "{t.val:.3f} ({t.avg:.3f})".format(t=meter['acc']),
                            "{t.val:.3f} ({t.avg:.3f})".format(t=meter['loss'])
                            )
        logger.info(log)

    return meter['loss'].avg, meter['acc'].avg, meter['batch_time']


if __name__ == '__main__':
    """Argument Parsing"""
    parser = argparse.ArgumentParser("Phone Finder")
    parser.add_argument('--mode', type=str, default=opt.mode)
    parser.add_argument('--arch', type=str, default=opt.arch)
    parser.add_argument('--model_dir', type=str, default=opt.model_dir)
    parser.add_argument('--data_dir', type=str, default=opt.data_dir)

    # Training hyperparameters
    parser.add_argument('--rotation', dest=CONST.ROTATION, action='store_true',
                       default=opt.rotation)
    parser.add_argument('--lr', type=float, default=opt.lr)
    parser.add_argument('--epochs', type=int, default=opt.epochs)
    parser.add_argument('--batch_size', '-b', type=int, default=opt.batch_size)
    parser.add_argument('--freezed_layers', type=int, default=opt.freezed_layers)
    parser.add_argument('--pretrained', dest=CONST.PRETRAINED, action='store_true',
                        default=opt.pretrained)

    # Training flags
    parser.add_argument('--gpu', '-g', type=str, default=opt.gpu)
    parser.add_argument('--resume', dest=CONST.RESUME, default=opt.resume, action='store_true')
    parser.add_argument('--print_freq', type=int, default=opt.print_freq)
    parser.add_argument('--log2file', dest=CONST.LOG2FILE, action='store_true',
                        default=opt.log2file)
    parser.add_argument('--logging_level', type=int, default=opt.logging_level)
    
    # # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # Example of passing in arguments as the new configurations
    #TODO find more efficient way to pass in arguments into configuration file
    mode = arguments.pop(CONST.MODE).lower()
    arch = arguments.pop(CONST.ARCH)
    model_dir = arguments.pop(CONST.MODEL_DIR)
    data_dir = arguments.pop(CONST.DATA_DIR)
    rotation = arguments.pop(CONST.ROTATION)
    lr = arguments.pop(CONST.LR)
    epochs = arguments.pop(CONST.EPOCHS)
    batch_size = arguments.pop(CONST.BATCH)
    freezed_layers = arguments.pop(CONST.FREEZED_LAYERS)
    gpu = arguments.pop(CONST.GPU)
    resume = arguments.pop(CONST.RESUME)
    print_freq = arguments.pop(CONST.PRINT_FREQ)
    log2file = arguments.pop(CONST.LOG2FILE)
    logging_level = arguments.pop(CONST.LOGGING_LVL)
    pretrained = arguments.pop(CONST.PRETRAINED)

    opt = set_config(mode=mode, arch=arch,
                     model_dir=model_dir, data_dir=data_dir, rotation=rotation,
                     lr=lr, epochs=epochs, batch_size=batch_size,
                     freezed_layers=freezed_layers,
                     gpu=gpu, resume=resume,
                     print_freq=print_freq, log2file=log2file,
                     logging_level=logging_level, pretrained=pretrained)
    
    # Initialize Logger
    log_fname = '{}.log'.format(opt.mode)
    Logger(log_filename=os.path.join(opt.model_dir, log_fname),
                    level=opt.logging_level, log2file=opt.log2file)
    logger = logging.getLogger('go-train')
    Logger.section_break('User Config')
    logger.info(pformat(opt._state_dict()))

    # Train and evaluate
    train_and_evaluate(opt, logger)

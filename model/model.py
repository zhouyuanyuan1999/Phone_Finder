import logging

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, resnet18, resnet34

from utils.logger import Logger
from utils.config import opt

VGG16 = 'vgg16'
RESNET18 = 'resnet18'
RESNET34 = 'resnet34'

class MODEL(nn.Module):
    __names__ = {RESNET18, RESNET34, VGG16}

    def __init__(self, arch, num_classes, pretrained=False):
        super(MODEL, self).__init__()

        assert arch in MODEL.__names__

        self.pretrained = pretrained
        self.num_class = num_classes
        self.logger = logging.getLogger(__name__)
        
        if arch == VGG16:
            self.feature_extractor, self.classifier = MODEL.get_vgg16_arch(
                num_classes, pretrained)

        elif arch == RESNET18:
            self.feature_extractor, self.classifier = MODEL.get_resnet18_arch(
                num_classes, pretrained)

        elif arch == RESNET34:
            self.feature_extractor, self.classifier = MODEL.get_resnet34_arch(
                num_classes, pretrained)
            
        Logger.section_break('Model')
        self.logger.info(f'Architecture selected: {arch} | Pretrained: {pretrained} | '
                         f'Num classes: {num_classes}')

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def get_vgg16_arch(num_classes, pretrained=False):
        vgg16_model = vgg16_bn(pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(vgg16_model.features.children()))

        num_features = vgg16_model.classifier[-1].in_features
        classifier = nn.Sequential(*list(vgg16_model.classifier.children())[:-1],
                                   nn.Linear(num_features, num_classes))

        return feature_extractor, classifier

    @staticmethod
    def get_resnet18_arch(num_classes, pretrained=False):
        resnet18_model = resnet18(pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-1])

        num_features = resnet18_model.fc.in_features
        
        classifier = nn.Linear(num_features, num_classes, bias=True)

        return feature_extractor, classifier

    @staticmethod
    def get_resnet34_arch(num_classes, pretrained=False):
        resnet34_model = resnet34(pretrained=pretrained)

        feature_extractor = nn.Sequential(*list(resnet34_model.children())[:-1])

        num_features = resnet34_model.fc.in_features
        classifier = nn.Linear(num_features, num_classes)

        return feature_extractor, classifier
    
def freezing_layers(model):
    """ Used to freeze layers of the given model
    Args:
        model (ResNet): ResNet model
    Returns:
        None
    """
    logger = logging.getLogger('Model_freeze')
    idx = 0
    for name, param in model.named_parameters():
        idx += 1
        logger.info(name + ' ('+str(idx)+')')
    logger.info('There are total '+str(idx)+'layers.')
    logger.info('Number of Freezed Layers ' + str(opt.freezed_layers))
    idx = 0
    last = False
    for name, param in model.named_parameters():
        idx += 1
        if idx <= opt.freezed_layers:
            param.requires_grad = False
        else:
            param.requires_grad = True
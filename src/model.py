import os
import torch
import math
from torch import nn
import torch.nn.functional as F
from torchvision import models
from package import *
from efficientnet_pytorch import EfficientNet

# from src.seresnet import se_resnext101, GeM
# from src.base.resnext_wsl import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl
# from src.inceptionresnetv2 import inceptionresnetv2

def get_torchvision_model(net_type, is_trained, num_classes, loss):
    """ Get torchvision model

    Parameters
    ----------
    net_type: str
        deep network type
    is_trained: boolean
        use pretrained ImageNet
    num_classes: int
        number of classes

    Returns
    -------
    nn.Module
        model based on net_type
    """
    if net_type.startswith("resnext"):
        return Resnext(net_type, num_classes, loss)
    elif net_type.startswith("efficientnet"):
        return CustomEfficientNet(net_type, is_trained, num_classes, loss)
    elif net_type.startswith("inceptionresnet"):
        return Inceptionnetv2(num_classes, loss)
    elif net_type.startswith("resnet"):
        return  Resnet(net_type, num_classes, loss, is_trained)
    return InceptionV3(is_trained, num_classes, loss)

class CustomEfficientNet(nn.Module):
    def __init__(self, net_type, is_trained, num_classes, loss):
        super().__init__()
        self.net = EfficientNet.from_pretrained(net_type) if is_trained else EfficientNet.from_name(net_type)
        kernel_count = self.net._fc.in_features
        self.net._fc = nn.Linear(kernel_count, num_classes)
        if loss == "focal":
            self.net._fc.bias.data.fill_(-math.log((1-0.01)/0.01))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.net(x)
        if not self.training:
             x = self.sigmoid(x)
        return x

class InceptionV3(nn.Module):
    def __init__(self, is_trained, num_classes, loss):
        super().__init__()
        self.net = models.inception_v3(pretrained=is_trained)
        aux_in = self.net.AuxLogits.fc.in_features
        self.net.AuxLogits.fc = nn.Sequential(nn.Linear(aux_in, num_classes))
        kernel_count = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(kernel_count, num_classes))
        if loss == "focal":
            self.net.fc.bias.data.fill_(-math.log((1-0.01)/0.01))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.net(x)
        if not self.training:
             x = self.sigmoid(x)
        return x
        
class Resnext(nn.Module):
    """
    ResNeXt-101 32x8d: net_type = " resnext101_32x8d_wsl "
    ResNeXt-101 32x16d: net_type = " resnext101_32x16d_wsl "
    ResNeXt-101 32x32d: net_type = " resnext101_32x32d_wsl "	
    ResNeXt-101 32x48d: net_type = " resnext101_32x48d_wsl "

    """

    def __init__(self, net_type, num_classes, loss):
        super().__init__()
        self.net = torch.hub.load('facebookresearch/WSL-Images', net_type)

        ### Change avg_pool by GeM
        self.net.avg_pool = GeM()
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        ### set initial bias value for using focal loss
        if loss == "focal":
            self.net.fc.bias.data.fill_(-math.log((1-0.01)/0.01))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x =self.net(x)
        if not self.training:
            x = self.sigmoid(x)
        return x

class Resnet(nn.Module):
    """
    net_type: resnet18, resnet34, resnet50, resnet101, resnet152

    """
    def __init__(self, net_type, num_classes, loss, is_trained):
        super().__init__()
        if net_type.endswith("18"):
            self.net = resnet18(pretrained=is_trained)
        elif net_type.endswith("34"):
            self.net = resnet34(pretrained=is_trained)
        elif net_type.endswith("50"):
            self.net = resnet50(pretrained=is_trained)
        elif net_type.endswith("101"):
            self.net = resnet101(pretrained=is_trained)
        else: 
            self.net = resnet152(pretrained=is_trained)
        
        ### Change avg_pool by GeM
        self.net.avg_pool = GeM()
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        ### set bias value for using focal loss
        if loss == "focal":
            self.net.fc.bias.data.fill_(-math.log((1-0.01)/0.01))

        self.sigmoid = nn.Sigmoid()

        #convert ReLU to swish:
        convert_relu_to_swish(self.net)

    def forward(self, x):
        x =self.net(x)
        if not self.training:
            x = self.sigmoid(x)
        return x

class Inceptionnetv2(nn.Module):
    def __init__(self, num_label, loss):
        super().__init__()
        self.net = inceptionresnetv2(pretrained='imagenet')
        self.net.avgpool_1a = GeM()
        self.net.last_linear= nn.Linear(self.net.last_linear.in_features,  num_label)
        if loss == "focal":
            self.net.last_linear.bias.data.fill_(-math.log((1-0.01)/0.01))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x =self.net(x)
        if not self.training:
            x = self.sigmoid(x)
        return x

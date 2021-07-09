import torch.nn as nn
import torch
from torchvision import models

from misc.layer import Conv2d, FC

import torch.nn.functional as F
from misc.utils import *
from .CSRNet import CSRNet
from .Res50 import Res50
import pdb
import pretrainedmodels

# model_path = '../PyTorch_Pretrained/resnet50-19c8e357.pth'

class Seresnet50(nn.Module):
    def __init__(self,  pretrained=True):
        super(Seresnet50, self).__init__()


        se_resnet50 = pretrainedmodels.se_resnet50()
        self.ser = nn.Sequential(se_resnet50.layer0, se_resnet50.layer1, se_resnet50.layer2)

        self.de_pred1 = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))
      

        self.de_pred2 = nn.Sequential(Conv2d(512, 512, 1, same_padding=True, NL='relu'),
                                     Conv2d(512, 1, 1, same_padding=True, NL='relu'))





    def forward(self,x):


        x = self.ser(x)


        x_de_pred1 = self.de_pred1(x)


        x_de_pred2 = self.de_pred2(x)


        x = (x_de_pred1 + x_de_pred2)/2
        x = F.upsample(x, scale_factor=8)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)   




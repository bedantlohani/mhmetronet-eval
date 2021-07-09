from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *


# model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'

class Dens121(nn.Module):
    def __init__(self, pretrained=True):
        super(Dens121, self).__init__()
        dens = models.densenet121(pretrained=True)

        l = dens.features
        self.features = nn.Sequential(l.conv0 , l.norm0, l.relu0, l.pool0, l.denseblock1, l.transition1, l.denseblock2)


        self.de_pred1 = nn.Sequential(Conv2d(512, 256, 1, same_padding=True, NL='relu'),
                                     Conv2d(256, 1, 1, same_padding=True, NL='relu'))

        self.de_pred2 = nn.Sequential(Conv2d(512, 64, 1, same_padding=True, NL='relu'),
                                     Conv2d(64, 1, 1, same_padding=True, NL='relu'))


        self.de_pred3 = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))

        self.de_pred4 = nn.Sequential(Conv2d(512, 512, 1, same_padding=True, NL='relu'),
                                     Conv2d(512, 1, 1, same_padding=True, NL='relu'))



    def forward(self, x):
        #print(x.shape)
        x = self.features(x)
        x1 = self.de_pred1(x)
        x2 = self.de_pred2(x)
        x3 = self.de_pred3(x)
        x4 = self.de_pred4(x)
        x = (x1 + x2 + x3 + x4)/4
        #print(x.shape)
        x = F.upsample(x,scale_factor=8)

        return x

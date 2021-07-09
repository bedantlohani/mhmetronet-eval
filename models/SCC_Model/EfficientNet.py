import torch.nn as nn
import torch
from torchvision import models

from misc.layer import Conv2d, FC

import torch.nn.functional as F
from misc.utils import *

import pdb
from efficientnet_pytorch import EfficientNet
model_path = '../PyTorch_Pretrained/resnet101-5d3b4d8f.pth'


class EffNetb5(nn.Module):
    def __init__(self, pretrained=True):
        super(EffNetb5, self).__init__()

        effNet =EfficientNet.from_pretrained('efficientnet-b5')
        # modello1 =nn.Sequential(*list(effNet._conv_stem.children()))
        modello2 = nn.Sequential(*list(effNet._bn0.children()))

        modello3 = nn.Sequential(*list(effNet._blocks.children())[:10])

        self.frontend = nn.Sequential(effNet._conv_stem, effNet._bn0, modello3)


        self.de_pred1 = nn.Sequential(Conv2d(64, 256, 3, same_padding=True, NL='relu'),
                                     Conv2d(256, 32, 3, same_padding=True, NL='relu'))
      

        self.de_pred2 = nn.Sequential(Conv2d(64, 64, 3, same_padding=True, NL='relu'),
                                     Conv2d(64, 32, 3, same_padding=True, NL='relu'))

        self.de_pred3 = nn.Sequential(Conv2d(64, 256, 5, same_padding=True, NL='relu'),
                                     Conv2d(256, 32, 5, same_padding=True, NL='relu'))

        self.de_pred4 = nn.Sequential(Conv2d(64, 64, 5, same_padding=True, NL='relu'),
                                     Conv2d(64, 32, 5, same_padding=True, NL='relu'))

        self.output = nn.Sequential(Conv2d(32,1,1,same_padding=True,NL='relu'))


        initialize_weights(self.modules())



    def forward(self, x):

        x = self.frontend(x)

        #x = self.own_reslayer_3(x)

        x1 = self.de_pred1(x)
        x2 = self.de_pred2(x)
        x3 = self.de_pred3(x)
        x4 = self.de_pred4(x)

        x = (x1 + x2 + x3 + x4 )/4
        x = self.output(x)

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





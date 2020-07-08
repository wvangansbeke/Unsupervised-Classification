"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch.nn as nn
import torchvision.models as models


def resnet50():
    backbone = models.__dict__['resnet50']()
    backbone.fc = nn.Identity()
    return {'backbone': backbone, 'dim': 2048}

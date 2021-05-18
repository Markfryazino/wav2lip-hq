import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class Enhancer(nn.Module):
    def __init__(self):
        self.backbone = nn.Sequential([
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 32, kernel_size=3, stride=1, padding=1, residual=True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=1),
            nn.Sigmoid()
        ])

    def forward(self, X):
        return self.backbone(X)

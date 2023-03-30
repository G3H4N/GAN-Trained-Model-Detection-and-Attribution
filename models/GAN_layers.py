from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from torch import nn


# ==============================================================================
# =                                    layer                                   =
# ==============================================================================

class NoOp(nn.Module):

    def __init__(self, *args, **keyword_args):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class Reshape(nn.Module):

    def __init__(self, *new_shape):
        super(Reshape, self).__init__()
        self._new_shape = new_shape

    def forward(self, x):
        new_shape = (x.size(i) if self._new_shape[i] == 0 else self._new_shape[i] for i in range(len(self._new_shape)))
        return x.view(*new_shape)


# ==============================================================================
# =                                layer wrapper                               =
# ==============================================================================

def identity(x, *args, **keyword_args):
    return x


# ==============================================================================
# =                                    DCGAN                                   =
# ==============================================================================

class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)

class CBR2D(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=4, padding=1, norm=True, relu=[], bias=False):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm:
            layers += [nn.BatchNorm2d(nch_out)]
        if relu != []:
            layers += [ReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self,x):
        return self.cbr(x)

class DECBR2D(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, norm=True, relu=[], bias=False):
        super().__init__()

        layers = []
        layers += [nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm:
            layers += [nn.BatchNorm2d(nch_out)]

        if relu != []:
            layers += [ReLU(relu)]

        self.decbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.decbr(x)
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

ACT_FN = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.2),
}


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channel: int,
            filter_channel: int,
            kernel_size: int,
            stride: int = 2,
            n_input_dims: int = 5,
            padding: int | str = 1,
            bias: bool = False,
            act_fn: str = 'relu',
    ):
        super(ConvBlock, self).__init__()

        if n_input_dims == 5:
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif n_input_dims == 4:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        self.act_fn = ACT_FN[act_fn]
        self.conv_bn_1 = nn.Sequential(
            conv(in_channel, filter_channel // 4, kernel_size, stride, padding=padding, bias=bias),
            bn(filter_channel // 4),
        )
        self.conv_bn_2 = nn.Sequential(
            conv(filter_channel // 4, filter_channel, kernel_size, padding=padding, bias=bias),
            bn(filter_channel),
        )
        self.conv_bn_sc = nn.Sequential(
            conv(in_channel, filter_channel, 1, stride, padding=0, bias=bias),
            bn(filter_channel),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # Conv branch
        x = self.conv_bn_1(inputs)
        x = self.act_fn(x)
        x = self.conv_bn_2(x)
        # Shortcut branch
        shortcut = self.conv_bn_sc(inputs)
        # Residual connection
        x = x + shortcut
        x = self.act_fn(x)

        return x


class IdentityBlock(nn.Module):
    def __init__(
            self,
            in_channel: int,
            filter_channel: int,
            kernel_size: int,
            n_input_dims: int = 5,
            padding: int | str = 1,
            bias: bool = False,
            act_fn: str = 'relu',
    ):
        super(IdentityBlock, self).__init__()

        if n_input_dims == 5:
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif n_input_dims == 4:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        self.act_fn = ACT_FN[act_fn]
        self.conv_bn_1 = nn.Sequential(
            conv(in_channel, filter_channel // 4, kernel_size, padding=padding, bias=bias),
            bn(filter_channel // 4),
        )
        self.conv_bn_2 = nn.Sequential(
            conv(filter_channel // 4, filter_channel, kernel_size, padding=padding, bias=bias),
            bn(filter_channel),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # Conv branch
        x = self.conv_bn_1(inputs)
        x = self.act_fn(x)
        x = self.conv_bn_2(x)

        # Residual connection
        x = x + inputs
        x = self.act_fn(x)

        return x

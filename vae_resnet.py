import torch
import numpy as np
import pickle
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

from module import ConvBlock, IdentityBlock


def sampling(x: Tuple[Tensor, Tensor]) -> Tensor:
    mu, log_var = x
    eps = torch.randn(mu.shape, device=mu.device)

    return mu + torch.exp(log_var / 2) * eps


def transform_ed(density) -> Tensor:
    """Transform electron density"""
    density = torch.log(density + 1e-4)
    return density / torch.log(1e-4)


def preprocess(data: Tensor) -> Tensor:
    return transform_ed(torch.tanh(data))


class VAEEncoder(nn.Module):
    def __init__(
            self,
            input_channel: int,
            encoder_conv_filter_channels: List[int],
            encoder_conv_kernel_sizes: List[int],
            encoder_conv_strides: List[int],
            z_dim: int,
            r_loss_factor: float,
            use_conv_bias: bool = False,
            use_batch_norm: bool = False,
            use_dropout: bool = False,
    ):
        super().__init__()

        self.encoder_conv_kernels = encoder_conv_filter_channels
        self.encoder_conv_kernel_size = encoder_conv_kernel_sizes
        self.encoder_conv_strides = encoder_conv_strides
        self.z_dim = z_dim
        self.r_loss_factor = r_loss_factor
        self.use_conv_bias = use_conv_bias
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.n_layers_encoder = len(encoder_conv_filter_channels)

        resnet = []
        for i in range(self.n_layers_encoder):
            # Get the input params of every conv layer
            in_channel = input_channel if i == 0 else self.encoder_conv_kernels[i - 1]
            filter_channel = self.encoder_conv_kernels[i]
            kernel_size = self.encoder_conv_kernel_size[i]
            strides = self.encoder_conv_strides[i]

            # Create the conv block
            resnet.append(ConvBlock(
                in_channel,
                filter_channel,
                kernel_size,
                strides,
                padding=1,
                bias=self.use_conv_bias,
            ))
            # Create the identity block
            resnet.append(IdentityBlock(
                filter_channel,
                filter_channel,
                kernel_size,
                strides,
                padding=1,
                bias=self.use_conv_bias,
            ))
        self.resnet = nn.Sequential(*resnet)
        del resnet
        self.proj_mu = nn.Linear(4*4*4*128, self.z_dim)
        self.proj_logvar = nn.Linear(4*4*4*128, self.z_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        mu = self.proj_mu(x)
        log_var = self.proj_logvar(x)
        z = sampling((mu, log_var))

        return mu, log_var, z

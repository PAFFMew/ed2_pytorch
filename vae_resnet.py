import torch
import numpy as np
import pickle
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


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
            encoder_conv_filters: List[int],
            encoder_conv_kernel_size: List[int],
            encoder_conv_strides: List[int],
            z_dim: int,
            r_loss_factor: float,
            use_batch_norm: bool = False,
            use_dropout: bool = False,
    ):
        super().__init__()

        self.encoder_conv_kernels = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.z_dim = z_dim
        self.r_loss_factor = r_loss_factor

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        for i in range(self.n_layers_encoder):
            # Get the input params of every conv layer
            filters = self.encoder_conv_kernels[i]
            kernel_size = self.encoder_conv_kernel_size[i]
            strides = self.encoder_conv_strides[i]
            # Init conv block in Resnet50


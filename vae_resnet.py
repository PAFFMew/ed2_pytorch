import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import math

from module import ConvBlock, IdentityBlock


def sampling(x: Tuple[Tensor, Tensor]) -> Tensor:
    mu, log_var = x
    eps = torch.randn(mu.shape, device=mu.device)

    return mu + torch.exp(log_var / 2) * eps


def transform_ed(density) -> Tensor:
    """Transform electron density"""
    density = torch.log(density + 1e-4)
    return density / math.log(1e-4)


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
            n_input_dims: int = 5,
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
                n_input_dims=n_input_dims,
                padding=1,
                bias=self.use_conv_bias,
            ))
            # Create the identity block
            resnet.append(IdentityBlock(
                filter_channel,
                filter_channel,
                kernel_size,
                n_input_dims=n_input_dims,
                padding=1,
                bias=self.use_conv_bias,
            ))
        self.resnet = nn.Sequential(*resnet)
        del resnet
        self.proj_mu = nn.Linear(4*4*4*128, self.z_dim)
        self.proj_logvar = nn.Linear(4*4*4*128, self.z_dim)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.resnet(inputs)
        x = x.view(x.size(0), -1)
        mu = self.proj_mu(x)
        log_var = self.proj_logvar(x)
        z = sampling((mu, log_var))

        return mu, log_var, z


class VAEDecoder(nn.Module):
    def __init__(
            self,
            output_channel: int,
            decoder_conv_filter_channels: List[int],
            decoder_conv_kernel_sizes: List[int],
            decoder_conv_strides: List[int],
            z_dim: int,
            r_loss_factor: float,
            n_output_dims: int = 5,
            use_conv_bias: bool = False,
            use_batch_norm: bool = False,
            use_dropout: bool = False,
    ):
        super().__init__()

        self.decoder_conv_kernels = decoder_conv_filter_channels
        self.decoder_conv_kernel_size = decoder_conv_kernel_sizes
        self.decoder_conv_strides = decoder_conv_strides
        self.z_dim = z_dim
        self.r_loss_factor = r_loss_factor
        self.use_conv_bias = use_conv_bias
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.n_layers_decoder = len(decoder_conv_filter_channels)

        resnet = []
        for i in range(self.n_layers_decoder):
            # Get the input params of every conv layer
            out_channel = output_channel if i == self.n_layers_decoder-1 else self.decoder_conv_kernels[i + 1]
            filter_channel = self.decoder_conv_kernels[i]
            kernel_size = self.decoder_conv_kernel_size[i]
            strides = self.decoder_conv_strides[i]

            # in the decoder we will upsample instead of using conv strides to downsample
            resnet.append(*[torch.nn.Upsample(scale_factor=2) for _ in range(strides - 1)])
            # Create the conv block
            resnet.append(ConvBlock(
                filter_channel,
                out_channel,
                kernel_size,
                1,
                n_input_dims=n_output_dims,
                padding=1,
                bias=self.use_conv_bias,
            ))
            # Create the identity block
            resnet.append(IdentityBlock(
                out_channel,
                out_channel,
                kernel_size,
                n_input_dims=n_output_dims,
                padding=1,
                bias=self.use_conv_bias,
            ))
        self.resnet = nn.Sequential(*resnet)
        del resnet
        self.proj_z = nn.Linear(self.z_dim, 4*4*4*128)
        # last one with 1 feature map
        self.proj_out = ConvBlock(
            self.decoder_conv_kernels[-1],
            output_channel,
            self.decoder_conv_kernel_size[-1],
            stride=1,
            n_input_dims=n_output_dims,
            padding=1,
            bias=self.use_conv_bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.proj_z(inputs)
        x = x.view(x.size(0), -1, 4, 4, 4)
        x = self.resnet(x)
        x = self.proj_out(x)

        return x


class VAE(nn.Module):
    def __init__(
            self,
            encoder_input_channel: int,
            decoder_output_channel: int,
            encoder_conv_filter_channels: List[int],
            decoder_conv_filter_channels: List[int],
            encoder_conv_kernel_sizes: List[int],
            decoder_conv_kernel_sizes: List[int],
            encoder_conv_strides: List[int],
            decoder_conv_strides: List[int],
            z_dim: int,
            r_loss_factor: float,
            n_input_dims: int = 5,
            n_output_dims: int = 5,
            use_conv_bias: bool = False,
            use_batch_norm: bool = False,
            use_dropout: bool = False,
    ):
        super().__init__()

        self.encoder = VAEEncoder(
            encoder_input_channel,
            encoder_conv_filter_channels,
            encoder_conv_kernel_sizes,
            encoder_conv_strides,
            z_dim,
            r_loss_factor,
            n_input_dims,
            use_conv_bias,
            use_batch_norm,
            use_dropout,
        )
        self.decoder = VAEDecoder(
            decoder_output_channel,
            decoder_conv_filter_channels,
            decoder_conv_kernel_sizes,
            decoder_conv_strides,
            z_dim,
            r_loss_factor,
            n_output_dims,
            use_conv_bias,
            use_batch_norm,
            use_dropout,
        )
        self._init_params()

    def _init_params(self, mode: str = 'orthogonal'):
        """Init all params of weights and biases."""
        init_methods = {
            'orthogonal': nn.init.orthogonal_,
            'kaiming': nn.init.kaiming_uniform_,
        }
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                init_methods[mode](m.weight)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = preprocess(inputs)
        mu_z, logvar_z, latent_z = self.encoder(x)
        x = self.decoder(latent_z)

        return x

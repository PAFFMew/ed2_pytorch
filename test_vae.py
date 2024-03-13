from torchsummary import summary
from vae import VAEResnet

from helper import load_config


def main():
    # 1. Give params directly
    # vae = VAEResnet(
    #     encoder_input_channel=1,
    #     decoder_output_channel=16,
    #     encoder_conv_filter_channels=[16, 32, 64, 128],
    #     decoder_conv_filter_channels=[128, 64, 32, 16],
    #     encoder_conv_kernel_sizes=[3, 3, 3, 3],
    #     decoder_conv_kernel_sizes=[3, 3, 3, 3],
    #     encoder_conv_strides=[2, 2, 2, 2],
    #     decoder_conv_strides=[2, 2, 2, 2],
    #     z_dim=400,
    #     n_input_dims=5,
    #     n_output_dims=5,
    #     use_conv_bias=True,
    #     use_batch_norm=True,
    #     use_dropout=True,
    # )

    # 2. Give params from config (more convenient)
    config = load_config()['model']['vae']
    vae = VAEResnet(**config)

    vae.cuda()
    summary(vae, (1, 64, 64, 64))


if __name__ == "__main__":
    main()

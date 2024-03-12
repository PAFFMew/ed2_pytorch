from torchsummary import summary
from vae_resnet import VAEEncoder


def main():
    # 1. Test VAE encoder
    encoder = VAEEncoder(
        input_channel=1,
        encoder_conv_filter_channels=[16, 32, 64, 128],
        encoder_conv_kernel_sizes=[3, 3, 3, 3],
        encoder_conv_strides=[2, 2, 2, 2],
        z_dim=400,
        r_loss_factor=50000,
        use_conv_bias=True,
        use_batch_norm=True,
        use_dropout=True,
    )
    encoder.cuda()
    summary(encoder, (1, 64, 64, 64))


if __name__ == "__main__":
    main()

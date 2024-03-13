from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Mapping
import lightning as pl
from functools import partial

from vae import VAEResnet, preprocess


class QM9PretrainWrapper(pl.LightningModule):
    def __init__(
            self,
            config: Dict | None = None,
    ):
        super(QM9PretrainWrapper, self).__init__()

        if config is None:
            from helper import load_config
            self.config = load_config()
        else:
            self.config = config
        self.vae = VAEResnet(**self.config['model']['vae'])
        self.r_loss_factor = self.config['model']['r_loss_factor']
        self.optimizer = self.config['model']['optimizer']
        # Define the initial random seed
        self._set_seed()

    def _set_seed(self):
        pl.seed_everything(self.config['seed'])

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step in lightning module.

        Note:
            This method is an override class method, and its initial params contain `batch` and `batch_idx`.
            `batch_idx` has not been used here, but we still need to keep it in the method signature.
        """
        x, y = batch
        x = preprocess(x)
        mu, logvar, latent_z = self.vae.encoder(x)
        out = self.vae.decoder(latent_z)
        total_loss, reconstruction_loss, kl_loss = self._loss(
            (x, out, mu, logvar, latent_z),
            self.r_loss_factor
        )
        output = {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
        }

        return output

    def _loss(self, data: Tuple[torch.Tensor], r_loss_factor: float | int) -> Dict[Tensor, Tensor, Tensor]:
        """Loss function of QM9 pre-training process.

        Note:
            1. Components of losses
                - KL div loss
                - Reconstruct loss
            2. Components of `data`
                - x (torch.Tensor): The input of VAE encoder.
                - out (torch.Tensor): The output of VAE decoder.
                - mu (torch.Tensor): Mean value of datasets distribution; one of the output of VAE encoder.
                - logvar (torch.Tensor): Log of variance of datasets distribution; one of the output of VAE encoder.
                - latent_z (torch.Tensor): Latent space variable; one of the output of VAE encoder.
            3. Loss computation
                total_loss = kl_loss + r_loss_factor * reconstruction_loss

        Args:
            data (Tuple): The input of loss function, including several parts.

        Returns:
            kl_loss (torch.Tensor): The loss of KL divergence.
            reconstruction_loss (torch.Tensor): The MSE loss of molecule reconstruction.
            total_loss (torch.Tensor): The weighted sum of two loss functions.
        """
        x, out, mu, logvar, latent_z = data
        reconstruction_loss = F.mse_loss(x, out) * self.r_loss_factor
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        total_loss = kl_loss + reconstruction_loss
        return total_loss, reconstruction_loss, kl_loss

    def configure_optimizers(self) -> Mapping[str, Any]:
        """Configure the optimizer and scheduler in lightning module.

        Note:
            This method is an override class method, and it returns a dict containing optimizer and scheduler.

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer for model training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler for model training.
        """
        optimizers = {
            'Adam': partial(torch.optim.Adam, params=self.vae.parameters()),
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
        }

        optimizer = optimizers[self.optimizer](
            lr=self.config['model']['vae']['optimizer']['lr'],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

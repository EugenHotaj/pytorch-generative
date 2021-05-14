"""Implementation of the Variational Autoencoder [1] model.

TODO(ehotaj): Explain.

References (used throughout the code):
    [1]: https://arxiv.org/pdf/1312.6114.pdf
"""


import torch
from torch import nn

from pytorch_generative.models import base
from pytorch_generative.models.vae import vaes


class VAE(base.GenerativeModel):
    """The Variational Autoencoder model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        latent_channels=16,
        strides=[4],
        hidden_channels=64,
        residual_channels=32,
    ):
        """Initializes a new VAE instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            latent_channels: Number of channels for each latent variable.
            strides: List of encoder/decoder strides. For each stride, we create an
                encoder (decoder) which downsamples (upsamples) the input by the stride.
            hidden_channels: Number of channels in (non residual block) hidden layers.
            residual_channels: Number of hidden channels in residual blocks.
        """
        super().__init__()

        self._latent_channels = latent_channels
        self._total_stride = sum(strides)

        encoder = []
        for i, stride in enumerate(strides):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = (
                hidden_channels if i < len(strides) - 1 else 2 * self._latent_channels
            )
            encoder.append(
                vaes.Encoder(
                    in_channels=in_c,
                    out_channels=out_c,
                    hidden_channels=hidden_channels,
                    residual_channels=residual_channels,
                    n_residual_blocks=2,
                    stride=stride,
                )
            )
        self._encoder = nn.Sequential(*encoder)

        decoder = []
        for i, stride in enumerate(reversed(strides)):
            in_c = self._latent_channels if i == 0 else hidden_channels
            out_c = hidden_channels if i < len(strides) - 1 else out_channels
            decoder.append(
                vaes.Decoder(
                    in_channels=in_c,
                    out_channels=out_c,
                    hidden_channels=hidden_channels,
                    residual_channels=residual_channels,
                    n_residual_blocks=2,
                    stride=stride,
                )
            )
        self._decoder = nn.Sequential(*decoder)

    def forward(self, x):
        """Computes the forward pass.

        Args:
            x: Batch of inputs.
        Returns:
            Tuple of the forward pass result and the total KL Divergence between the
            prior and the posterior. Note that the KL Divergence is NOT normalized by
            the dimension of the input.
        """
        # NOTE: We output log_std both for numerical stability and to ensure that
        # the variance is positive since log_std.exp().pow(2) >= 0.
        mean, log_std = torch.split(self._encoder(x), self._latent_channels, dim=1)
        kl_div = vaes.unit_gaussian_kl_div(mean, log_std).sum(dim=(1, 2, 3))
        latents = vaes.sample_from_gaussian(mean, log_std)
        return self._decoder(latents), kl_div

    def sample(self, n_samples):
        """Generates a batch of n_samples."""
        latent_size = self._h // 2 ** (self._total_stride // 2)
        shape = (n_samples, self._latent_channels, latent_size, latent_size)
        latents = torch.randn(shape, device=self.device)
        return self._decoder(latents)


def reproduce(
    n_epochs=457,
    batch_size=128,
    log_dir="/tmp/run",
    n_gpus=1,
    device_id=0,
    debug_loader=None,
):
    """Training script with defaults to reproduce results.

    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.

    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        n_gpus: Number of GPUs to use for training the model. If 0, uses CPU.
        device_id: The device_id of the current GPU when training on multiple GPUs.
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """
    import torch
    from torch import optim
    from torch.nn import functional as F

    from pytorch_generative import datasets
    from pytorch_generative import models
    from pytorch_generative import trainer

    train_loader, test_loader = debug_loader, debug_loader
    if train_loader is None:
        train_loader, test_loader = datasets.get_mnist_loaders(
            batch_size, dynamically_binarize=True, resize_to_32=True
        )

    model = models.VAE(
        in_channels=1,
        out_channels=1,
        latent_channels=16,
        strides=[2, 2, 2, 2],
        hidden_channels=64,
        residual_channels=32,
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    def loss_fn(x, _, preds):
        preds, kl_div = preds
        recon_loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
        recon_loss = recon_loss.sum(dim=(1, 2, 3))
        elbo = recon_loss + kl_div

        return {
            "recon_loss": recon_loss.mean(),
            "kl_div": kl_div.mean(),
            "loss": elbo.mean(),
        }

    def sample_fn(model):
        sample = torch.sigmoid(model.sample(n_samples=16))
        return torch.where(
            sample < 0.5, torch.zeros_like(sample), torch.ones_like(sample)
        )

    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        sample_epochs=1,
        sample_fn=sample_fn,
        log_dir=log_dir,
        n_gpus=n_gpus,
        device_id=device_id,
    )
    model_trainer.interleaved_train_and_eval(n_epochs)

"""Implementation of the VQ-VAE [1] model.

The Vector Quantized Variational Autoencoder (VQ-VAE) extends the vanilla 
Variational Autoencoder by mapping the input to a discrete (instead of 
continuous) latent space. The discretization is performed using vector 
quantization where continuous outputs from an encoder are mapped to the nearest
vectors in a learned codebook. The main intuition behind using discretized
latents is that the underlying domain we are trying to model is usually 
discrete (e.g. words in a sentence, objects in an image, etc).

References (used throughout the code):
    [1]: https://arxiv.org/pdf/1711.00937.pdf
"""

from torch import nn

from pytorch_generative.models import base
from pytorch_generative.models.vae import vaes


class VectorQuantizedVAE(base.GenerativeModel):
    """The Vector Quantized Variational Autoencoder (VQ-VAE) model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_channels=128,
        n_residual_blocks=2,
        residual_channels=32,
        n_embeddings=128,
        embedding_dim=16,
    ):
        """Initializes a new VectorQuantizedVAE instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Number of channels in (non residual block) hidden layers.
            n_residual_blocks: Number of residual blocks in each residual stack.
            residual_channels: Number of hidden channels in residual blocks.
            n_embeddings: Number of VectorQuantizer embeddings.
            embedding_dim: Dimension of the VectorQuantizer embeddings.
        """
        super().__init__()
        self._encoder = vaes.Encoder(
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_residual_blocks=n_residual_blocks,
            residual_channels=residual_channels,
            stride=4,
        )
        self._quantizer = vaes.Quantizer(
            in_channels=hidden_channels,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
        )
        self._decoder = vaes.Decoder(
            in_channels=embedding_dim,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_residual_blocks=n_residual_blocks,
            residual_channels=residual_channels,
            stride=4,
        )

    def forward(self, x):
        """Computes the forward pass.

        Args:
            x: Batch of inputs.
        Returns:
            Tuple of the forward pass result and the quantization loss.
        """
        x = self._encoder(x)
        quantized, quantization_loss = self._quantizer(x)
        return self._decoder(quantized), quantization_loss

    def sample(self, n_samples):
        raise NotImplementedError("VQ-VAE does not support sampling.")


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
    from torch import optim
    from torch.nn import functional as F
    from torch.optim import lr_scheduler

    from pytorch_generative import datasets
    from pytorch_generative import models
    from pytorch_generative import trainer

    train_loader, test_loader = debug_loader, debug_loader
    if train_loader is None:
        train_loader, test_loader = datasets.get_cifar10_loaders(
            batch_size, normalize=True
        )

    model = models.VectorQuantizedVAE(
        in_channels=3,
        out_channels=3,
        hidden_channels=128,
        residual_channels=32,
        n_residual_blocks=2,
        n_embeddings=512,
        embedding_dim=64,
    )
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.999977)

    def loss_fn(x, _, preds):
        preds, vq_loss = preds
        recon_loss = F.mse_loss(preds, x)
        loss = recon_loss + vq_loss

        return {
            "vq_loss": vq_loss,
            "reconstruction_loss": recon_loss,
            "loss": loss,
        }

    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        lr_scheduler=scheduler,
        log_dir=log_dir,
        n_gpus=n_gpus,
        device_id=device_id,
    )
    model_trainer.interleaved_train_and_eval(n_epochs)

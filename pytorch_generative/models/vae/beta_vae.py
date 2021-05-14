"""Implementation of the Beta-VAE [1] model.

Beta-VAE extends the vanilla VAE by introducing a scaling factor `Beta` for the KL 
Divergence term in the Evidence Lower Bound (ELBO) loss. For priors with mutually 
independent factors, such as an Isotropic Gaussian, this has the effect of regularizing
the learned latent variables to be conditionally independent. The motivation is that 
independent latents will learn distangled representations of the dataset.

References (used throughout the code):
    [1]: https://openreview.net/pdf?id=Sy2fzU9gl 
"""

from pytorch_generative.models.vae import vae


class BetaVAE(vae.VAE):
    """The Beta-VAE model.

    NOTE: This implementation is merely provided for convenience. The same effect can
    easily be achieved by scaling the KL Divergence value returned by the vanilla VAE.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        beta=4.0,
        latent_channels=16,
        strides=[4],
        hidden_channels=64,
        residual_channels=32,
    ):
        """Initializes a new BetaVAE instance.

        Args:
            in_channels: See the base class.
            out_channels: See the base class.
            beta: Scaling factor applied to the KL Divergence between the approximate
                posterior and the prior. `beta=1.0` corresponds to the vanilla VAE.
            latent_channels: See the base class.
            strides: See the base class.
            hidden_channels: See the base class.
            residual_channels: See the base class.
        """
        super().__init__(
            in_channels,
            out_channels,
            latent_channels,
            strides,
            hidden_channels,
            residual_channels,
        )
        self._beta = beta

    def forward(self, x):
        out, kl_div = super().forward(x)
        return out, self._beta * kl_div


def reproduce(
    n_epochs=500,
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

    model = models.BetaVAE(
        in_channels=1,
        out_channels=1,
        beta=4.0,
        latent_channels=16,
        strides=[2, 2, 2, 2],
        hidden_channels=64,
        residual_channels=32,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

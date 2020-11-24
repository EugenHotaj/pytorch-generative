"""Implementation of PixelCNN [1].

PixelCNN extends Masked Autoregressive Density Estimation (MADE) [2] to 
convolutional neural networks. Convolutional filters are masked to respect the
autoregressive property so that the outputs of each filter only depend on left
and above inputs (see MaskedConv2d for details).

NOTE: Our implementation does *not* use autoregressive channel masking. This
means that each output depends on whole pixels and not sub-pixels. For outputs
with multiple channels, other methods can be used, e.g. [3].

[1]: https://arxiv.org/abs/1601.06759
[2]: https://arxiv.org/abs/1502.03509
[2]: https://arxiv.org/abs/1701.05517
"""

import torch
from torch import distributions
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base


class MaskedResidualBlock(nn.Module):
    """A residual block masked to respect the autoregressive property."""

    def __init__(self, n_channels):
        """Initializes a new MaskedResidualBlock instance.

        Args:
            n_channels: The number of input (and output) channels.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1
            ),
            nn.ReLU(),
            pg_nn.MaskedConv2d(
                is_causal=False,
                in_channels=n_channels // 2,
                out_channels=n_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels // 2, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self._net(x)


class PixelCNN(base.AutoregressiveModel):
    """The PixelCNN model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_residual=15,
        residual_channels=128,
        head_channels=32,
        sample_fn=None,
    ):
        """Initializes a new PixelCNN instance.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            n_residual: The number of residual blocks.
            residual_channels: The number of channels to use in the residual layers.
            head_channels: The number of channels to use in the two 1x1 convolutional
                layers at the head of the network.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._input = pg_nn.MaskedConv2d(
            is_causal=True,
            in_channels=in_channels,
            out_channels=2 * residual_channels,
            kernel_size=7,
            padding=3,
        )
        self._masked_layers = nn.ModuleList(
            [
                MaskedResidualBlock(n_channels=2 * residual_channels)
                for _ in range(n_residual)
            ]
        )
        self._head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2 * residual_channels,
                out_channels=head_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=head_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        x = self._input(x)
        for layer in self._masked_layers:
            x = x + layer(x)
        return self._head(x)


def reproduce(
    n_epochs=457, batch_size=256, log_dir="/tmp/run", device="cuda", debug_loader=None
):
    """Training script with defaults to reproduce results.

    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.

    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        device: Device to train on (either 'cuda' or 'cpu').
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """
    from torch import optim
    from torch.nn import functional as F
    from torch.optim import lr_scheduler
    from torch.utils import data
    from torchvision import datasets
    from torchvision import transforms

    from pytorch_generative import trainer
    from pytorch_generative import models

    transform = transforms.Compose(
        [transforms.ToTensor(), lambda x: distributions.Bernoulli(probs=x).sample()]
    )
    train_loader = debug_loader or data.DataLoader(
        datasets.MNIST("/tmp/data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    test_loader = debug_loader or data.DataLoader(
        datasets.MNIST("/tmp/data", train=False, download=True, transform=transform),
        batch_size=batch_size,
        num_workers=8,
    )

    model = models.PixelCNN(
        in_channels=1,
        out_channels=1,
        n_residual=15,
        residual_channels=16,
        head_channels=32,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.999977)

    def loss_fn(x, _, preds):
        batch_size = x.shape[0]
        x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
        loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
        return loss.sum(dim=1).mean()

    trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        lr_scheduler=scheduler,
        log_dir=log_dir,
        device=device,
    )
    trainer.interleaved_train_and_eval(n_epochs)

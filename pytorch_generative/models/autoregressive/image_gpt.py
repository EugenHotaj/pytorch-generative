"""Implementation of (convolutional) ImageGPT [1].

ImageGPT is an autoregressive model which applies the (decoder only) Transformer 
architecture to image generation.

NOTE: Our implementation operates over images instead of embedding tokens like [1]. This
defeats the purpose slightly as the main motivation of the original paper is to 
demonstrate that the same architecture can be effective for both images and text.

References (used throughout the file):
  [1]: https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf
"""

import torch
from torch import distributions
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base


class TransformerBlock(nn.Module):
    """An ImageGPT Transformer block."""

    def __init__(self, n_channels, n_attention_heads):
        """Initializes a new TransformerBlock instance.

        Args:
            n_channels: The number of input and output channels.
            n_attention_heads: The number of attention heads to use.
        """
        super().__init__()
        self._ln1 = pg_nn.NCHWLayerNorm(n_channels)
        self._ln2 = pg_nn.NCHWLayerNorm(n_channels)
        self._attn = pg_nn.CausalAttention(
            in_channels=n_channels,
            n_heads=n_attention_heads,
            embed_channels=n_channels,
            out_channels=n_channels,
        )
        self._out = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels, out_channels=4 * n_channels, kernel_size=1
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=4 * n_channels, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        x = x + self._attn(self._ln1(x))
        return x + self._out(self._ln2(x))


class ImageGPT(base.AutoregressiveModel):
    """The ImageGPT Model.

    Unlike [1], our implementation operates over image inputs, instead of
    embeddings. Furthermore, we implement skip connections from each block to the
    output. We find that this makes training a lot more stable and allows for much
    faster convergence.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        in_size=28,
        n_transformer_blocks=8,
        n_attention_heads=4,
        n_embedding_channels=16,
        sample_fn=None,
    ):
        """Initializes a new ImageGPT instance.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            in_size: Size of the input images. Used to create positional encodings.
            n_transformer_blocks: Number of TransformerBlocks to use.
            n_attention_heads: Number of attention heads to use.
            n_embedding_channels: Number of attention embedding channels to use.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._pos = nn.Parameter(torch.zeros(1, in_channels, in_size, in_size))
        self._input = pg_nn.CausalConv2d(
            mask_center=True,
            in_channels=in_channels,
            out_channels=n_embedding_channels,
            kernel_size=3,
            padding=1,
        )
        self._transformer = nn.ModuleList(
            TransformerBlock(
                n_channels=n_embedding_channels, n_attention_heads=n_attention_heads
            )
            for _ in range(n_transformer_blocks)
        )
        self._ln = pg_nn.NCHWLayerNorm(n_embedding_channels)
        self._out = nn.Conv2d(
            in_channels=n_embedding_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self._input(x + self._pos)
        for block in self._transformer:
            x = x + block(x)
        return self._out(self._ln(x))


def reproduce(
    n_epochs=457,
    batch_size=64,
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
        train_loader, test_loader = datasets.get_mnist_loaders(
            batch_size, dynamically_binarize=True
        )

    model = models.ImageGPT(
        in_channels=1,
        out_channels=1,
        in_size=28,
        n_transformer_blocks=8,
        n_attention_heads=2,
        n_embedding_channels=64,
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.999977)

    def loss_fn(x, _, preds):
        batch_size = x.shape[0]
        x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
        loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
        return loss.sum(dim=1).mean()

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

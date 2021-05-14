"""Implementation of the Gated PixelCNN [1].

Gated PixelCNN extends the original PixelCNN [2] by incorporating ideas 
motivated by the more effective PixelRNNs. The first extension is to use
GatedActivations (instead of ReLUs) to mimic the gated functions in RNN. The
second extension is to use a two-stream architecture to mitigate the blind spot
introduced by autoregressively masking convolution filters.

We follow the implementation in [3] but use a casually masked GatedPixelCNNLayer
for the input instead of a causally masked Conv2d layer. For efficiency, the 
masked Nx1 and 1xN convolutions are implemented via unmasked (N//2+1)x1 and
1x(N//2+1) convolutions with padding and cropping, as suggested in [1].

NOTE: Our implementaiton does *not* use autoregressive channel masking. This
means that each output depends on whole pixels not sub-pixels. For outputs with
multiple channels, other methods can be used, e.g. [4]. 

References (used throughout the code):
  [1]: https://arxiv.org/abs/1606.05328
  [2]: https://arxiv.org/abs/1601.06759
  [3]: http://www.scottreed.info/files/iclr2017.pdf
  [4]: https://arxiv.org/abs/1701.05517
"""

import torch
from torch import distributions
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base


class GatedPixelCNNLayer(nn.Module):
    """A Gated PixelCNN layer.

    The layer takes as input 'vstack' and 'hstack' from previous
    'GatedPixelCNNLayers' and returns 'vstack', 'hstack', 'skip' where 'skip' is
    the skip connection to the pre-logits layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False):
        """Initializes a new GatedPixelCNNLayer instance.

        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of output channels.
            kernel_size: The size of the (causal) convolutional kernel to use.
            mask_center: Whether the 'GatedPixelCNNLayer' is causal. If 'True', the
                center pixel is masked out so the computation only depends on pixels to
                the left and above. The residual connection in the horizontal stack is
                also removed.
        """
        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size cannot be even"

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activation = pg_nn.GatedActivation()
        self._kernel_size = kernel_size
        self._padding = (kernel_size - 1) // 2  # (kernel_size - stride) / 2
        self._mask_center = mask_center

        # Vertical stack convolutions.
        self._vstack_1xN = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=(1, self._kernel_size),
            padding=(0, self._padding),
        )
        # TODO(eugenhotaj): Is it better to shift down the the vstack_Nx1 output
        # instead of adding extra padding to the convolution? When we add extra
        # padding, the cropped output rows will no longer line up with the rows of
        # the vstack_1x1 output.
        self._vstack_Nx1 = nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=2 * self._out_channels,
            kernel_size=(self._kernel_size // 2 + 1, 1),
            padding=(self._padding + 1, 0),
        )
        self._vstack_1x1 = nn.Conv2d(
            in_channels=in_channels, out_channels=2 * out_channels, kernel_size=1
        )

        self._link = nn.Conv2d(
            in_channels=2 * out_channels, out_channels=2 * out_channels, kernel_size=1
        )

        # Horizontal stack convolutions.
        self._hstack_1xN = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=2 * self._out_channels,
            kernel_size=(1, self._kernel_size // 2 + 1),
            padding=(0, self._padding + int(self._mask_center)),
        )
        self._hstack_residual = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )
        self._hstack_skip = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, vstack_input, hstack_input):
        """Computes the forward pass.

        Args:
            vstack_input: The input to the vertical stack.
            hstack_input: The input to the horizontal stack.
        Returns:
            (vstack,  hstack, skip) where vstack and hstack are the vertical stack and
            horizontal stack outputs respectively and skip is the skip connection
            output.
        """
        _, _, h, w = vstack_input.shape  # Assuming NCHW.

        # Compute vertical stack.
        vstack = self._vstack_Nx1(self._vstack_1xN(vstack_input))[:, :, :h, :]
        link = self._link(vstack)
        vstack += self._vstack_1x1(vstack_input)
        vstack = self._activation(vstack)

        # Compute horizontal stack.
        hstack = link + self._hstack_1xN(hstack_input)[:, :, :, :w]
        hstack = self._activation(hstack)
        skip = self._hstack_skip(hstack)
        hstack = self._hstack_residual(hstack)
        # NOTE(eugenhotaj): We cannot use a residual connection for causal layers
        # otherwise we'll have access to future pixels.
        if not self._mask_center:
            hstack += hstack_input

        return vstack, hstack, skip


class GatedPixelCNN(base.AutoregressiveModel):
    """The Gated PixelCNN model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_gated=10,
        gated_channels=128,
        head_channels=32,
        sample_fn=None,
    ):
        """Initializes a new GatedPixelCNN instance.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            n_gated: The number of gated layers (not including the input layers).
            gated_channels: The number of channels to use in the gated layers.
            head_channels: The number of channels to use in the 1x1 convolution blocks
                in the head after all the gated channels.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._input = GatedPixelCNNLayer(
            in_channels=in_channels,
            out_channels=gated_channels,
            kernel_size=7,
            mask_center=True,
        )
        self._gated_layers = nn.ModuleList(
            [
                GatedPixelCNNLayer(
                    in_channels=gated_channels,
                    out_channels=gated_channels,
                    kernel_size=3,
                    mask_center=False,
                )
                for _ in range(n_gated)
            ]
        )
        self._head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=gated_channels, out_channels=head_channels, kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=head_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        vstack, hstack, skip_connections = self._input(x, x)
        for gated_layer in self._gated_layers:
            vstack, hstack, skip = gated_layer(vstack, hstack)
            skip_connections += skip
        return self._head(skip_connections)


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
        train_loader, test_loader = datasets.get_mnist_loaders(
            batch_size, dynamically_binarize=True
        )

    model = models.GatedPixelCNN(
        in_channels=1, out_channels=1, n_gated=10, gated_channels=128, head_channels=32
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.9999)

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

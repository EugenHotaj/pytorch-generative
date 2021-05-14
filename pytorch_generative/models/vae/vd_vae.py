"""Implementation of the Very Deep VAE [1] model.

TODO(eugenhotaj): explain.

References (used throughout code):
    [1]: https://arxiv.org/pdf/2011.10650.pdf
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pytorch_generative.models import base
from pytorch_generative.models.vae import vaes


@dataclass
class StackConfig:
    """Configuration for the encoder and decoder stacks at a given resolution.

    The Very Deep VAE model is an (inverted) U-Net architecture consisting of a number
    of encoding and decoding stacks. After each encoding stack, the input is downscaled
    by a factor of two. Similarly, after each decoding stack, the input is upscaled by
    a factor of two.

    During training, the activations from one encoding stack are fed as inputs to the
    corresponding decoding stack with same resolution.

    Note that n_encoder_blocks and n_decoder blocks can be different.
    """

    n_encoder_blocks: int
    n_decoder_blocks: int


DEFAULT_MODEL = [
    StackConfig(n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(n_encoder_blocks=1, n_decoder_blocks=1),
]


class BottleneckBlock(nn.Module):
    """A (potentially residual) bottleneck block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        bottleneck_kernel_size=3,
        is_residual=True,
    ):
        """Initializes a new BottleneckBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bottleneck_channels: Number of channels in middle (bottleneck) convolutions.
            bottleneck_kernel_size: Kernel size for middle (bottleneck) convolutions.
            is_residual: Whether to use a residual connection from input to the output.
        """
        super().__init__()
        self._is_residual = is_residual

        # TODO(eugenhotaj): This only works for kernel_size in {1, 3}.
        padding = 1 if bottleneck_kernel_size == 3 else 0
        self._net = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_size=bottleneck_kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_size=bottleneck_kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=bottleneck_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        h = self._net(x)
        return x + h if self._is_residual else h


class TopDownBlock(nn.Module):
    """Top-down block as introduced in [1]."""

    def __init__(
        self,
        n_channels,
        latent_channels,
        bottleneck_channels,
        bottleneck_kernel_size,
    ):
        """Initializes a new TopDownBlock.

        Args:
            n_channels: Number of input/output channels.
            latent_channels: Number of channels in the latent code.
            bottleneck_channels: Number of bottleneck channels for BottleneckBlocks.
            bottleneck_kernel_size: Bottleneck kernel size for BottleneckBlocks.
        """
        super().__init__()
        self._n_channels = n_channels
        self._latent_channels = latent_channels

        self._prior = BottleneckBlock(
            in_channels=self._n_channels,
            out_channels=2 * self._latent_channels + self._n_channels,
            bottleneck_channels=bottleneck_channels,
            is_residual=False,
        )
        self._posterior = BottleneckBlock(
            in_channels=2 * self._n_channels,
            out_channels=2 * self._latent_channels,
            bottleneck_channels=bottleneck_channels,
            is_residual=False,
        )
        self._latents = nn.Conv2d(
            in_channels=self._latent_channels,
            out_channels=self._n_channels,
            kernel_size=1,
        )
        self._out = BottleneckBlock(
            in_channels=self._n_channels,
            out_channels=self._n_channels,
            bottleneck_channels=bottleneck_channels,
            bottleneck_kernel_size=bottleneck_kernel_size,
            is_residual=True,
        )

    def forward(self, x, mixin=None):
        """Computes the forward pass.

        Args:
            x: Batch of inputs.
            mixin: Activations from the bottom up pass which are used to compute the
                approximate posterior. If not 'None' the latents are sampled from the
                approximated posterior, otherwise the approximate posterior is not
                computed and the latents are sampled from the prior.
        Returs:
            A tuple (activations, kl_div) where kl_div is the KL divergence between the
            approximate posterior and the prior if mixin is not None, or None otherwise.
        """
        p_mean, p_log_std, p_h = torch.split(
            self._prior(x),
            [self._latent_channels, self._latent_channels, self._n_channels],
            dim=1,
        )

        # If mixin is None, we must be in the generation regime, so we sample 'z' from
        # the prior. Otherwise, we are in the the training regime, so we sample 'z' from
        # the (approximate) posterior.
        if mixin is None:
            z = vaes.sample_from_gaussian(p_mean, p_log_std)
            kl_div = None
        else:
            q_mean, q_log_std = torch.split(
                self._posterior(torch.cat((x, mixin), dim=1)),
                self._latent_channels,
                dim=1,
            )
            z = vaes.sample_from_gaussian(q_mean, q_log_std)
            kl_div = vaes.gaussian_kl_div(q_mean, q_log_std, p_mean, p_log_std)

        latents = self._latents(z)
        return self._out(x + p_h + latents), kl_div


class EncoderStack(nn.Module):
    """An encoding module comprised of a stack of ResidualBlocks."""

    def __init__(
        self,
        n_residual_blocks,
        pool,
        n_channels,
        bottleneck_channels,
        bottleneck_kernel_size,
    ):
        """Initializes a new EncoderStack.

        Args:
            n_residual_blocks: Number of residual blocks in the EncoderStack.
            pool: Whether to average pool the output before returning it.
            n_channels: Number of input/output channels.
            bottleneck_channels: Number of bottleneck channels for BottleneckBlocks.
            bottleneck_kernel_size: Bottleneck kernel size for BottleneckBlocks.
        """
        super().__init__()
        residuals = [
            BottleneckBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                bottleneck_channels=bottleneck_channels,
                bottleneck_kernel_size=bottleneck_kernel_size,
                is_residual=True,
            )
            for _ in range(n_residual_blocks)
        ]
        self._residuals = nn.Sequential(*residuals)
        self._pool = nn.AvgPool2d(kernel_size=2, stride=2) if pool else None

    def forward(self, x):
        features = self._residuals(x)
        x = self._pool(features) if self._pool is not None else features
        return x, features


class DecoderStack(nn.Module):
    """A decoding module comprised of a stack of TopDownBlocks."""

    def __init__(
        self,
        n_topdown_blocks,
        unpool,
        n_channels,
        latent_channels,
        bottleneck_channels,
        bottleneck_kernel_size,
    ):
        """Initializes a new DecoderStack.

        Args:
            n_topdown_blocks: Number of TopDownBlocks in the DecoderStack.
            unpool: Whether to nearest-neighbor unpool the input before using it.
            n_channels: Number of input/output channels.
            latent_channels: Number of channels in the latent code.
            bottleneck_channels: Number of bottleneck channels for BottleneckBlocks.
            bottleneck_kernel_size: Bottleneck kernel size for BottleneckBlocks.
        """

        super().__init__()
        self._unpool = nn.Upsample(scale_factor=2, mode="nearest") if unpool else None
        topdowns = [
            TopDownBlock(
                n_channels,
                latent_channels,
                bottleneck_channels,
                bottleneck_kernel_size,
            )
            for _ in range(n_topdown_blocks)
        ]
        self._topdowns = nn.ModuleList(topdowns)

    def forward(self, x, mixin=None):
        """Computes the forward pass.

        Args:
            x: Batch of inputs.
            mixin: Activations from bottom up pass. See TopDownBlock for more details.
        Returs:
            A tuple (activations, kl_divs) where kl_divs is a list of the KL divergences
            returned by all the TopDownBlocks in the stack.
        """
        if self._unpool is not None:
            x = self._unpool(x)
        kl_divs = []
        for topdown in self._topdowns:
            x, kl_div = topdown(x, mixin)
            kl_divs.append(kl_div)
        return x, kl_divs


class VeryDeepVAE(base.GenerativeModel):
    """The Very Deep VAE Model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_resolution=32,
        stack_configs=DEFAULT_MODEL,
        latent_channels=4,
        hidden_channels=16,
        bottleneck_channels=8,
    ):
        """Initializes a new VeryDeepVAE instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            input_resolution: Initial resolution of the input image. The input is
                downsampled by a factor of two after each encoder stack. See StackConfig
                for more details.
            stack_configs: A list of StackConfigs defining the model architecture.
            latent_channels: Number of channels in the latent code.
            hidden_channels: Number of non-bottleneck channels.
            bottleneck_channels: Number of bottleneck channels.
        """
        super().__init__()

        # Encoder.
        self._input = nn.Conv2d(
            in_channels, out_channels=hidden_channels, kernel_size=3, padding=1
        )
        self._encoder = nn.ModuleList()
        resolutions = [input_resolution // 2 ** i for i in range(len(stack_configs))]
        encoder_blocks = [conf.n_encoder_blocks for conf in stack_configs]
        total_encoder_blocks = sum(encoder_blocks)
        for i, (res, n_blocks) in enumerate(zip(resolutions, encoder_blocks)):
            pool = i < len(stack_configs) - 1
            bottleneck_kernel_size = 3 if res >= 3 else 1
            stack = EncoderStack(
                n_residual_blocks=n_blocks,
                pool=pool,
                n_channels=hidden_channels,
                bottleneck_channels=bottleneck_channels,
                bottleneck_kernel_size=bottleneck_kernel_size,
            )
            # Scale weights of res blocks' last conv by 1 / sqrt(n_blocks).
            for block in stack._residuals:
                block._net[-1].weight.data /= np.sqrt(total_encoder_blocks)
            self._encoder.append(stack)

        # NOTE: Bias tensors are used as input into to the decoder during training. We
        # can then later use these bias tensors to sample from the model.
        biases = [
            nn.Parameter(torch.zeros(1, hidden_channels, size, size))
            for size in resolutions[1:] + [resolutions[-1]]
        ]
        self._biases = nn.ParameterList(biases)

        # Decoder.
        self._decoder = nn.ModuleList()
        decoder_blocks = [conf.n_decoder_blocks for conf in stack_configs]
        total_decoder_blocks = sum(decoder_blocks)
        reverse = reversed(resolutions), reversed(decoder_blocks)
        for i, (res, n_blocks) in enumerate(zip(*reverse)):
            unpool, bottlneck_kernel_size = i > 0, 3 if res >= 3 else 1
            stack = DecoderStack(
                n_topdown_blocks=n_blocks,
                unpool=unpool,
                n_channels=hidden_channels,
                latent_channels=latent_channels,
                bottleneck_channels=bottleneck_channels,
                bottleneck_kernel_size=bottleneck_kernel_size,
            )
            # Scale weights of res block's last conv and latents by 1 / sqrt(n_blocks).
            for block in stack._topdowns:
                block._out._net[-1].weight.data /= np.sqrt(total_decoder_blocks)
                block._latents.weight.data /= np.sqrt(total_decoder_blocks)
            self._decoder.append(stack)
        self._output = nn.Conv2d(
            in_channels=hidden_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        """Computes the forward pass.

        Args:
            x: Batch of inputs.
        Returns:
            Tuple of the forward pass result and the total KL Divergence between the
            prior and the posterior. Note that the KL Divergence is NOT normalized by
            the dimension of the input.
        """
        n = x.shape[0]

        # Bottom up encoding.
        x = self._input(x)
        mixins = []
        for stack in self._encoder:
            x, mixin = stack(x)
            mixins.append(mixin)

        # Top down decoding.
        x = torch.zeros_like(self._biases[-1]).repeat(n, 1, 1, 1)
        kl_divs = []
        zipped = zip(self._decoder, reversed(mixins), reversed(self._biases))
        for stack, mixin, bias in zipped:
            x += bias.repeat(n, 1, 1, 1)
            x, divs = stack(x, mixin)
            kl_divs.extend(divs)

        # Compute total KL Divergence.
        kl_div = torch.zeros((n,), device=self.device)
        for div in kl_divs:
            kl_div += div.sum(dim=(1, 2, 3))

        return self._output(x), kl_div

    def sample(self, n_samples):
        x = torch.zeros_like(self._biases[-1]).repeat(n_samples, 1, 1, 1)
        for stack, bias in zip(self._decoder, reversed(self._biases)):
            x += bias.repeat(n_samples, 1, 1, 1)
            x, _ = stack(x)
        return self._output(x)


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

    stack_configs = [
        StackConfig(n_encoder_blocks=3, n_decoder_blocks=5),
        StackConfig(n_encoder_blocks=3, n_decoder_blocks=5),
        StackConfig(n_encoder_blocks=2, n_decoder_blocks=4),
        StackConfig(n_encoder_blocks=2, n_decoder_blocks=3),
        StackConfig(n_encoder_blocks=2, n_decoder_blocks=2),
        StackConfig(n_encoder_blocks=1, n_decoder_blocks=1),
    ]

    model = models.VeryDeepVAE(
        in_channels=1,
        out_channels=1,
        input_resolution=32,
        stack_configs=stack_configs,
        latent_channels=16,
        hidden_channels=64,
        bottleneck_channels=32,
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

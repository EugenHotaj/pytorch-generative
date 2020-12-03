"""TODO."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class StackConfig:
    resolution: int
    n_encoder_blocks: int
    n_decoder_blocks: int


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        bottleneck_kernel_size=3,
        is_residual=True,
    ):
        super().__init__()
        self._is_residual = is_residual

        # TODO(eugenhotaj): This only works for kernel_size=3.
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
    def __init__(
        self,
        n_channels,
        latent_channels,
        bottleneck_channels,
        bottleneck_kernel_size,
    ):
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

    def _kl_div(self, q_mean, q_log_std, p_mean, p_log_std):
        # NOTE: This KL divergence is only applicable under the assumption that both
        # the prior and the posterior are Isotropic Gaussian distributions.
        q_var, p_var = q_log_std.exp() ** 2, p_log_std.exp() ** 2
        mean_delta, std_delta = (q_mean - p_mean) ** 2, p_log_std - q_log_std
        return -0.5 + std_delta + 0.5 * (q_var + mean_delta) / p_var

    def _z(self, mean, log_std):
        return mean + log_std.exp() * torch.randn_like(log_std)

    def forward(self, x, mixin=None):
        p_mean, p_log_std, p_h = torch.split(
            self._prior(x),
            [self._latent_channels, self._latent_channels, self._n_channels],
            dim=1,
        )

        # If mixin is None, we must be in the generation regime, so we sample 'z' from
        # the prior. Otherwise, we are in the the training regime, so we sample 'z' from
        # the (approximate) posterior.
        if mixin is None:
            z = self._z(p_mean, p_log_std)
            kl_div = None
        else:
            q_mean, q_log_std = torch.split(
                self._posterior(torch.cat((x, mixin), dim=1)),
                self._latent_channels,
                dim=1,
            )
            z = self._z(q_mean, q_log_std)
            kl_div = self._kl_div(q_mean, q_log_std, p_mean, p_log_std)

        latents = self._latents(z)
        return self._out(x + p_h + latents), kl_div


class EncoderStack(nn.Module):
    def __init__(
        self,
        n_residual_blocks,
        pool,
        n_channels,
        bottleneck_channels,
        bottleneck_kernel_size,
    ):
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
    def __init__(
        self,
        n_topdown_blocks,
        unpool,
        n_channels,
        latent_channels,
        bottleneck_channels,
        bottleneck_kernel_size,
    ):
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

    def forward(self, x, bottom_up=None):
        if self._unpool is not None:
            x = self._unpool(x)
        kl_divs = []
        for topdown in self._topdowns:
            x, kl_div = topdown(x, bottom_up)
            kl_divs.append(kl_div)
        return x, kl_divs


default_model = [
    StackConfig(resolution=32, n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(resolution=16, n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(resolution=8, n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(resolution=4, n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(resolution=2, n_encoder_blocks=1, n_decoder_blocks=1),
    StackConfig(resolution=1, n_encoder_blocks=1, n_decoder_blocks=1),
]


class VeryDeepVAE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        stack_configs=default_model,
        latent_channels=4,
        hidden_channels=16,
        bottleneck_channels=8,
    ):
        super().__init__()

        # Encoder.
        self._input = nn.Conv2d(
            in_channels, out_channels=hidden_channels, kernel_size=3, padding=1
        )
        self._encoder = nn.ModuleList()
        resolutions = [conf.resolution for conf in stack_configs]
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
            # Initialize weights of last conv in each residual block to 1 / n_blocks.
            for block in stack._residuals:
                block._net[-1].weight.data *= np.sqrt(1 / total_encoder_blocks)
            self._encoder.append(stack)

        # NOTE: We feed bias tensors into the decoder so we can later sample.
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
            # Initialize weights of last conv in each topdown block to 1 / n_blocks.
            for block in stack._topdowns:
                block._out._net[-1].weight.data *= np.sqrt(1 / total_decoder_blocks)
            self._decoder.append(stack)
        self._output = nn.Conv2d(
            in_channels=hidden_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        n, c, h, w = x.shape

        # Bottom up encoding.
        x = self._input(x)
        mixins = []
        for stack in self._encoder:
            x, mixin = stack(x)
            mixins.append(mixin)

        # Top down decoding.
        x = torch.zeros_like(self._biases[-1]).repeat(n, 1, 1, 1)
        kl_divs = []
        to_zip = self._decoder, reversed(mixins), reversed(self._biases)
        for stack, mixin, bias in zip(*to_zips):
            x += bias.repeat(n, 1, 1, 1)
            x, divs = stack(x, mixin)
            kl_divs.extend(divs)

        # Compute total KL Divergence.
        kl_div = torch.zeros((n,), device=x.device)
        n_dims = np.prod((c, h, w))
        for div in kl_divs:
            kl_div += div.sum(dim=(1, 2, 3))
        kl_div /= n_dims

        return self._output(x), kl_div

    def sample(self, n_samples):
        x = torch.zeros_like(self._biases[-1]).repeat(n_samples, 1, 1, 1)
        for stack, bias in zip(self._decoder, reversed(self._biases)):
            x += bias.repeat(n_samples, 1, 1, 1)
            x, _ = stack(x)
        return self._output(x)

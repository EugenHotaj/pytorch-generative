"""Common modules used by Variational Autoencoders."""

import numpy as np
import torch
from torch import nn

from pytorch_generative import nn as pg_nn


@torch.jit.script
def to_var(log_std):
    """Returns the variance given the log standard deviation."""
    return log_std.exp().pow(2)


@torch.jit.script
def unit_gaussian_kl_div(mean, log_std):
    """Returns `KL(p || N(0, 1))` where `p` is a Gaussian with diagonal covariance."""
    return -0.5 * (1 + 2 * log_std - to_var(log_std) - mean ** 2)


@torch.jit.script
def gaussian_kl_div(p_mean, p_log_std, q_mean, q_log_std):
    """Returns `KL(p || q)` where `p` and `q` are Gaussians with diagonal covariance."""
    mean_delta, log_std_delta = (p_mean - q_mean) ** 2, q_log_std - p_log_std
    p_var, q_var = to_var(p_log_std), 2 * to_var(q_log_std)
    return -0.5 + log_std_delta + (p_var + mean_delta) / q_var


@torch.jit.script
def sample_from_gaussian(mu, log_sig):
    """Returns a sample from a Gaussian with diagonal covariance."""
    return mu + log_sig.exp() * torch.randn_like(log_sig)


@torch.jit.script
def _unflatten_tril(x):
    """Unflattens a vector into a lower triangular matrix of shape `dim x dim`."""
    n, dim = x.shape
    idxs = torch.tril_indices(dim, dim)
    tril = torch.zeros(n, dim, dim)
    tril[:, idxs[0, :], idxs[1, :]] = x
    return tril


@torch.jit.script
def gaussian_log_prob(x, mu, chol_sig):
    """Returns the log likelihood of `x` under a Gaussian with full covariance.

    The covariance is assumed to be positive-definite and to have been decomposed into
    the Cholesky factor `chol_sig`, i.e. `sig = chol_sig @ chol_sig.T`.

    Args:
        x: The input for which to compute the log likelihood.
        mu: The mean of the multivariate Gaussian.
        chol_sig: The flattened lower-triangular Cholesky factor of the covariance.
    Returns:
        The log likelihood.
    """
    dim = x.shape[0]
    chol_sig = _unflatten_tril(chol_sig)
    sig = chol_sig @ chol_sig.T
    const = -0.5 * dim * torch.log(torch.tensor(2 * np.pi))
    log_det = -0.5 * torch.logdet(x)
    exp = -0.5 * ((x - mu).T @ sig.inverse() @ (x - mu))
    return const + log_det + exp


class ResidualBlock(nn.Module):
    """A simple residual block."""

    def __init__(self, n_channels, hidden_channels):
        """Initializes a new ResidualBlock instance.

        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self._net(x)


class ResidualStack(nn.Module):
    """A stack of multiple ResidualBlocks."""

    def __init__(self, n_channels, hidden_channels, n_residual_blocks=1):
        """Initializes a new ResidualStack instance.

        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
            n_residual_blocks: Number of residual blocks in the stack.
        """
        super().__init__()
        self._net = nn.Sequential(
            *[
                ResidualBlock(n_channels, hidden_channels)
                for _ in range(n_residual_blocks)
            ]
            + [nn.ReLU()]
        )

    def forward(self, x):
        return self._net(x)


class Encoder(nn.Module):
    """A feedforward encoder which downsamples its input."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        n_residual_blocks,
        residual_channels,
        stride,
    ):
        """Initializes a new Encoder instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Number of channels in non residual block hidden layers.
            n_residual_blocks: Number of residual blocks in each residual stack.
            residual_channels: Number of hidden channels in residual blocks.
            stride: Stride to use in the downsampling convolutions. Must be even.
        """
        super().__init__()
        assert stride % 2 == 0, '"stride" must be even.'

        net = []
        for i in range(stride // 2):
            first, last = 0, stride // 2 - 1
            in_c = in_channels if i == first else hidden_channels // 2
            out_c = hidden_channels // 2 if i < last else hidden_channels
            net.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            net.append(nn.ReLU())
        net.append(
            ResidualStack(
                n_channels=hidden_channels,
                hidden_channels=residual_channels,
                n_residual_blocks=n_residual_blocks,
            )
        )
        net.append(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self._net = nn.Sequential(*net)

    def forward(self, x):
        return self._net(x)


class Decoder(nn.Module):
    """A feedforward encoder which upsamples its input."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        n_residual_blocks,
        residual_channels,
        stride,
    ):
        """Initializes a new Decoder instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Number of channels in (non residual block) hidden layers.
            n_residual_blocks: Number of residual blocks in each residual stack.
            residual_channels: Number of hidden channels in residual blocks.
            stride: Stride to use in the upsampling (i.e. transpose) convolutions. Must
                be even.
        """
        super().__init__()

        assert stride % 2 == 0, '"stride" must be even.'

        net = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            ResidualStack(
                n_channels=hidden_channels,
                hidden_channels=residual_channels,
                n_residual_blocks=n_residual_blocks,
            ),
        ]
        for i in range(stride // 2):
            first, last = 0, stride // 2 - 1
            in_c = hidden_channels if i == first else hidden_channels // 2
            out_c = hidden_channels // 2 if i < last else out_channels
            net.append(
                nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if i < last:
                net.append(nn.ReLU())
        self._net = nn.Sequential(*net)

    def forward(self, x):
        return self._net(x)


class Quantizer(nn.Module):
    """Wraps a VectorQuantizer to handle input with arbitrary channels."""

    def __init__(self, in_channels, n_embeddings, embedding_dim):
        """Initializes a new Quantizer instance.

        Args:
            in_channels: Number of input channels.
            n_embeddings: Number of VectorQuantizer embeddings.
            embedding_dim: VectorQuantizer embedding dimension.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=embedding_dim, kernel_size=1
            ),
            pg_nn.VectorQuantizer(n_embeddings, embedding_dim),
        )

    def forward(self, x):
        return self._net(x)

"""Modules and functions for building attention models.

References (used throughout the code):
    [1]: https://arxiv.org/abs/1712.09763
    [2]: https://arxiv.org/abs/2006.16236
"""

import functools

import numpy as np
import torch
from torch import autograd
from torch import nn
from torch.nn import functional as F


@functools.lru_cache(maxsize=32)
def image_positional_encoding(shape):
    """Generates positional encodings for 2d images.

    The positional encoding is a Tensor of shape (N, 2, H, W) of (x, y) pixel
    coordinates scaled to be between -.5 and .5.

    Args:
        shape: NCHW shape of image for which to generate positional encodings.
    Returns:
        The positional encodings.
    """
    n, c, h, w = shape
    zeros = torch.zeros(n, 1, h, w)
    return torch.cat(
        (
            (torch.arange(-0.5, 0.5, 1 / h)[None, None, :, None] + zeros),
            (torch.arange(-0.5, 0.5, 1 / w)[None, None, None, :] + zeros),
        ),
        dim=1,
    )


@functools.lru_cache(maxsize=32)
def _get_causal_mask(size, mask_center):
    """Generates causal masks for attention weights."""
    return torch.tril(torch.ones((size, size)), diagonal=-int(mask_center))


class CausalAttention(nn.Module):
    """Autoregresively masked, multihead self-attention layer.

    Autoregressive masking means that the current pixel can only attend to itself,
    pixels to the left, and pixels above. When mask_center=True, the current pixel does
    not attent to itself.

    This Module generalizes attention to use 2D convolutions instead of fully connected
    layers. As such, the input is expected to be 4D image tensors.
    """

    def __init__(
        self,
        in_channels,
        n_heads=1,
        embed_channels=None,
        out_channels=None,
        mask_center=False,
        extra_input_channels=0,
    ):
        """Initializes a new CausalAttention instance.

        Args:
            in_channels: Number of input channels.
            n_heads: Number of causal self-attention heads.
            embed_channels: Number of embedding channels. Defaults to in_channels.
            out_channels: Number of output channels. Defaults to in_channels.
            extra_input_channels: Extra input channels which are only used to compute
                the embeddings and not the attention weights since doing so may break
                the autoregressive property. For example, in [1] these channels include
                the original input image.
            mask_center: Whether to mask the center pixel of the attention matrices.
        """
        super().__init__()
        self._n_heads = n_heads
        self._embed_channels = embed_channels or in_channels
        self._out_channels = out_channels or in_channels
        self._mask_center = mask_center

        self._q = nn.Conv2d(
            in_channels=in_channels, out_channels=self._embed_channels, kernel_size=1
        )
        self._kv = nn.Conv2d(
            in_channels=in_channels + extra_input_channels,
            out_channels=self._embed_channels + self._out_channels,
            kernel_size=1,
        )
        # TODO(eugenhotaj): Should we only project if n_heads > 1?
        self._proj = nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=self._out_channels,
            kernel_size=1,
        )

    def forward(self, x, extra_x=None):
        """Computes the forward pass.

        Args:
            x: The input used to compute both embeddings and attention weights.
            extra_x: Extra channels concatenated with 'x' only used to compute the
                embeddings. See the 'extra_input_channels' argument for more info.
        Returns:
            The result of the forward pass.
        """

        def _to_multihead(t):
            """Reshapes an (N, C, H, W) tensor into (N, n_heads, H * W, head_size)."""
            c = t.shape[1]
            t = t.view(n, self._n_heads, c // self._n_heads, -1)
            return t.transpose(2, 3)

        n, _, h, w = x.shape

        # Compute the query, key, and value.
        q = _to_multihead(self._q(x))
        if extra_x is not None:
            x = torch.cat((x, extra_x), dim=1)
        k, v = self._kv(x).split([self._embed_channels, self._out_channels], dim=1)
        k, v = _to_multihead(k), _to_multihead(v)

        # Compute the causual attention weights.
        mask = (
            _get_causal_mask(h * w, self._mask_center)
            .view(1, 1, h * w, h * w)
            .to(next(self.parameters()).device)
        )
        attn = (q @ k.transpose(2, 3)) / np.sqrt(k.shape[-1])
        attn = attn.masked_fill(mask == 0, -np.inf)
        attn = F.softmax(attn, dim=-1).masked_fill(mask == 0, 0)

        # Attent to output for each head, stack, and project.
        out = (attn @ v).transpose(2, 3).contiguous().view(n, -1, h, w)
        return self._proj(out)


def _idx(i):
    return (slice(None), slice(None), slice(i, i + 1, 1), slice(None))


class _UnnormalizedLinearCausalAttention(autograd.Function):
    """Computes unnormalized causal attention using only O(N*C) memory."""

    @staticmethod
    def forward(ctx, Q, K, V):
        ctx.save_for_backward(Q, K, V)

        Vnew, S = torch.zeros_like(V), 0
        for i in range(V.shape[2]):
            S = S + K[_idx(i)].transpose(2, 3) @ V[_idx(i)]
            Vnew[_idx(i)] = Q[_idx(i)] @ S
        return Vnew

    @staticmethod
    def backward(ctx, G):
        Q, K, V = ctx.saved_tensors

        dQ, S = torch.zeros_like(Q), 0
        for i in range(V.shape[2]):
            S = S + K[_idx(i)].transpose(2, 3) @ V[_idx(i)]
            dQ[_idx(i)] = G[_idx(i)] @ S.transpose(2, 3)

        dK, dV, S = torch.zeros_like(K), torch.zeros_like(V), 0
        for i in range(V.shape[2] - 1, -1, -1):
            S = S + Q[_idx(i)].transpose(2, 3) @ G[_idx(i)]
            dV[_idx(i)] = K[_idx(i)] @ S
            dK[_idx(i)] = V[_idx(i)] @ S.transpose(2, 3)
        return dQ, dK, dV


# TODO(eugenhotaj): LinearCausalAttention currently does O(N) computations each
# time forward is called. During sampling, forward is called N times to generate
# N pixels. This means that during sampling  LinearCausalAttention unnecessarily
# does O(N^2) computations, most of which are thrown away. Instead, we can do
# O(N) work during sampling by storing previous activations as proposed in [2].
# TODO(eugenhotaj): This API does not match the CausalAttention API. We need
# to add support for mask_center and extra_input. There is also a lot of shared
# code between the two which sould be extracted. It's probably possible to
# have base class which does the bookkeeping and the subclasses implement
# the actual computations.
class LinearCausalAttention(nn.Module):
    """Memory efficient implementation of CausalAttention as introduced in [2].

    NOTE: LinearCausalAttention is *much* slower than CausalAttention and should
    only be used if your model cannot fit in memory.

    This implementation only requiers O(N) memory (instead of O(N^2)) for a
    sequence of N elements (e.g. an image with N pixels). To achieve this memory
    reduction, the implementation avoids storing the full attention matrix in
    memory and instead computes the output directly as Q @ (K @ V). However, this
    output cannot be vectorized and requires iterating over the sequence, which
    drastically slows down the computation.
    """

    def __init__(
        self,
        in_channels,
        feature_fn=lambda x: F.elu(x) + 1,
        n_heads=1,
        embed_channels=None,
        out_channels=None,
    ):
        """Initializes a new LinearCausalAttention instance.

        Args:
            in_channels: Number of input channels.
            feature_fn: A kernel feature map applied to the Query and Key activations.
                Defaults to lambda x: elu(x) + 1.
            n_heads: Number of causal self-attention heads.
            embed_channels: Number of embedding channels. Defaults to in_channels.
            out_channels: Number of output channels. Defaults to in_channels.
        """
        super().__init__()
        self._feature_fn = feature_fn
        self._n_heads = n_heads
        self._embed_channels = embed_channels or in_channels
        self._out_channels = out_channels or in_channels

        self._query = nn.Conv2d(
            in_channels=in_channels, out_channels=self._embed_channels, kernel_size=1
        )
        self._kv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self._embed_channels + self._out_channels,
            kernel_size=1,
        )
        self._numerator = _UnnormalizedLinearCausalAttention.apply

    def forward(self, x):
        def _to_multihead(t):
            """Reshapes an (N, C, H, W) tensor into (N, n_heads, H * W, head_size)."""
            c = t.shape[1]
            t = t.view(n, self._n_heads, c // self._n_heads, -1)
            return t.transpose(2, 3)

        n, _, h, w = x.shape

        # Compute the Query, Key, and Value.
        Q = _to_multihead(self._query(x))
        K, V = self._kv(x).split([self._embed_channels, self._out_channels], dim=1)
        K, V = _to_multihead(K), _to_multihead(V)

        # Compute the causual attention weights.
        Q, K = self._feature_fn(Q), self._feature_fn(K)
        den = 1 / (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + 1e-10)
        num = self._numerator(Q, K, V)
        out = num * torch.unsqueeze(den, -1)
        return out.transpose(2, 3).contiguous().view(n, -1, h, w)

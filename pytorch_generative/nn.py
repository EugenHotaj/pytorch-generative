"""Modules, funcitons, and  building blocks for Generative Neural Networks.

References (used throughout the code):
  [1]: https://arxiv.org/abs/1601.06759
  [2]: https://arxiv.org/abs/1712.09763
  [3]: https://arxiv.org/abs/2006.16236
  [4]: https://arxiv.org/abs/1711.00937
"""

import functools

import numpy as np
import torch
from torch import autograd
from torch import nn
from torch.nn import functional as F
from torch.nn import init


@functools.lru_cache(maxsize=32)
def image_positional_encoding(shape):
  """Generates *per-channel* positional encodings for 2d images.

  The positional encoding is a Tensor of shape (N, 2*C, H, W) of (x, y) pixel 
  coordinates scaled to be between -.5 and .5. 

  Args: 
    shape: NCHW shape of image for which to generate positional encodings.
  Returns:
    The positional encodings.
  """
  n, c, h, w = shape
  zeros = torch.zeros(n, c, h, w) 
  return torch.cat((
    (torch.arange(-.5, .5, 1 / h)[None, None, :, None] + zeros),
    (torch.arange(-.5, .5, 1 / w)[None, None, None, :] + zeros)),
    dim=1)


class GatedActivation(nn.Module):
  """Activation function which computes actiation_fn(f) * sigmoid(g).
  
  The f and g correspond to the top 1/2 and bottom 1/2 of the input channels.
  """

  def __init__(self, activation_fn=torch.tanh):
    """Initializes a new GatedActivation instance.

    Args:
      activation_fn: Activation to use for the top 1/2 input channels.
    """
    super().__init__()
    self._activation_fn = activation_fn

  def forward(self, x):
    _, c, _, _ = x.shape
    assert c % 2 == 0, 'x must have an even number of channels.'
    x, gate = x[:, :c//2, :, :], x[:, c//2:, :, :]
    return self._activation_fn(x) * torch.sigmoid(gate)


class NCHWLayerNorm(nn.LayerNorm):
  """Applies LayerNorm to the channel dimension of NCHW tensors."""

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    x = super().forward(x)
    return x.permute(0, 3, 1, 2)


class MaskedConv2d(nn.Conv2d):
  """A Conv2d layer masked to respect the autoregressive property.

  Autoregressive masking means that the computation of the current pixel only
  depends on itself, pixels to the left, and pixels above. When the convolution
  is causally masked (i.e. is_causal=True), the computation of the current 
  pixel does not depend on itself.

  E.g. for a 3x3 kernel, the following masks are generated for each channel:
                      [[1 1 1],                   [[1 1 1],
      is_causal=False  [1 1 0],    is_causal=True  [1 0 0],
                       [0 0 0]]                    [0 0 0]
  In [1], they refer to the left masks as 'type A' and right as 'type B'. 

  NOTE: This layer does *not* implement autoregressive channel masking.
  """

  def __init__(self, is_causal, *args, **kwargs):
    """Initializes a new MaskedConv2d instance.
    
    Args:
      is_causal: Whether the convolution should be causally masked.
    """
    super().__init__(*args, **kwargs)
    i, o, h, w = self.weight.shape
    mask = torch.zeros((i, o, h, w))
    mask.data[:, :, :h//2, :] = 1
    mask.data[:, :, h//2, :w//2 + int(not is_causal)] = 1
    self.register_buffer('mask', mask)

  def forward(self, x):
    self.weight.data *= self.mask
    return super().forward(x)


@functools.lru_cache(maxsize=32)
def _get_causal_mask(size, is_causal):
  """Generates causal masks for attention weights."""
  return torch.tril(torch.ones((size, size)), diagonal=-int(is_causal))


class MaskedAttention(nn.Module):
  """Autoregresively masked multihead self-attention layer.

  Autoregressive masking means that the current pixel can only attend to itself,
  pixels to the left, and pixels above. When the convolution is causally masked
  (i.e. is_causal=True), the current pixel does not attent to itself.

  This Module generalizes attention to use 2D convolutions instead of fully 
  connected layers. As such, the input is expected to be 4D image tensors.
  """ 

  def __init__(self, 
               in_channels, 
               n_heads=1,
               embed_channels=None,
               out_channels=None,
               is_causal=False,
               extra_input_channels=0):
    """Initializes a new MaskedAttention instance.

    Args:
      in_channels: Number of input channels. 
      n_heads: Number of causal self-attention heads.
      embed_channels: Number of embedding channels. Defaults to in_channels.
      out_channels: Number of output channels. Defaults to in_channels.
      extra_input_channels: Extra input channels which are only used to compute
        the embeddings and not the attention weights since doing so may break
        the autoregressive property. For example, in [2] these channels include
        the original input image.
      is_causal: Whether the convolution should be causally masked.
    """
    super().__init__()
    self._n_heads = n_heads
    self._embed_channels = embed_channels or in_channels
    self._out_channels = out_channels or in_channels 
    self._is_causal = is_causal

    self._q = nn.Conv2d(in_channels=in_channels, 
                        out_channels=self._embed_channels, kernel_size=1)
    self._kv = nn.Conv2d(in_channels=in_channels + extra_input_channels,
                         out_channels=self._embed_channels + self._out_channels, 
                         kernel_size=1)
    # TODO(eugenhotaj): Should we only project if n_heads > 1?
    self._proj = nn.Conv2d(in_channels=self._out_channels, 
                           out_channels=self._out_channels,
                           kernel_size=1) 

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
    mask = _get_causal_mask(h * w, self._is_causal).view(1, 1, h * w, h * w).to(
        next(self.parameters()).device)
    attn = (q @ k.transpose(2, 3)) / np.sqrt(k.shape[-1])
    attn = attn.masked_fill(mask == 0, -np.inf)
    attn = F.softmax(attn, dim=-1).masked_fill(mask == 0, 0)

    # Attent to output for each head, stack, and project.
    out = (attn @ v).transpose(2, 3).contiguous().view(n, -1, h, w)
    return self._proj(out)


def _idx(i):
  return (slice(None), slice(None), slice(i, i+1, 1), slice(None))


class _UnnormalizedLinearMaskedAttention(autograd.Function):
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


# TODO(eugenhotaj): LinearMaskedAttention currently does O(N) computations each
# time forward is called. During sampling, forward is called N times to generate
# N pixels. This means that during sampling  LinearMaskedAttention unnecessarily
# does O(N^2) computations, most of which are thrown away. Instead, we can do
# O(N) work during sampling by storing previous activations as proposed in [3].
# TODO(eugenhotaj): This API does not match the MaskedAttention API. We need
# to add support for is_causal and extra_input. There is also a lot of shared
# code between the two which sould be extracted. It's probably possible to 
# have base class which does the bookkeeping and the subclasses implement
# the actual computations.
class LinearMaskedAttention(nn.Module):
  """Memory efficient implementation of MaskedAttention as introduced in [3].

  NOTE: LinearMaskedAttention is *much* slower than MaskedAttention and should
  only be used if your model cannot fit in memory.

  This implementation only requiers O(N) memory (instead of O(N^2)) for a
  sequence of N elements (e.g. an image with N pixels). To achieve this memory
  reduction, the implementation avoids storing the full attention matrix in
  memory and instead computes the output directly as Q @ (K @ V). However, this
  output cannot be vectorized and requires iterating over the sequence, which
  drastically slows down the computation.
  """

  def __init__(self,
               in_channels,
               feature_fn=lambda x: F.elu(x) + 1,
               n_heads=1,
               embed_channels=None,
               out_channels=None,
               is_causal=False):
    """Initializes a new MaskedAttention instance.

    Args:
      in_channels: Number of input channels.
      feature_fn: A kernel feature map applied to the Query and Key activations.
        Defaults to lambda x: elu(x) + 1.
      n_heads: Number of causal self-attention heads.
      embed_channels: Number of embedding channels. Defaults to in_channels.
      out_channels: Number of output channels. Defaults to in_channels.
      is_causal: Unused and always set to False..
    """
    super().__init__()
    self._feature_fn = feature_fn
    self._n_heads = n_heads
    self._embed_channels = embed_channels or in_channels
    self._out_channels = out_channels or in_channels

    self._query = nn.Conv2d(in_channels=in_channels,
                            out_channels=self._embed_channels, kernel_size=1)
    self._kv = nn.Conv2d(in_channels=in_channels,
                         out_channels=self._embed_channels + self._out_channels,
                         kernel_size=1)
    self._numerator = _UnnormalizedLinearMaskedAttention.apply

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


# TODO(eugenhotaj): It's strange that this module returns a loss.
class VectorQuantizer(nn.Module):
  """A vector quantizer as introduced in [4].
  
  Inputs are quantized to the closest embedding in Euclidian distance. The 
  embeddings can be updated using either exponential moving averages or gradient
  descent.
  """

  def __init__(self, n_embeddings, embedding_dim, use_ema=True, ema_decay=.99):
    """Initializes a new VectorQuantizer instance.
    
    Args:
      n_embeddings: The number of embedding vectors. Controls the capacity in 
        the information bottleneck.
      embedding_dim: Dimension of each embedding vector. Does not directly 
        affect the capacity in the information bottleneck.
      use_ema: Whether to use exponential moving averages (EMA) to update the
        embedding weights instead of gradient descent. Generally, EMA updates
        lead to much faster convergence.
      ema_decay: Decay rate for exponential moving average parameters.
    """
    super().__init__()
    self.n_embeddings = n_embeddings
    self.embedding_dim = embedding_dim
    self._use_ema = use_ema
    self._decay = ema_decay

    embedding = torch.zeros(n_embeddings, embedding_dim)
    # TODO(eugenhotaj): Small optimization: create pre-initialized embedding.
    init.kaiming_uniform_(embedding, nonlinearity='linear')
    if self._use_ema:
      self.register_buffer('_embedding', embedding) 
      self.register_buffer('_cluster_size', torch.zeros(n_embeddings))
      self.register_buffer('_embedding_avg', embedding.clone())
    else:
      self._embedding = nn.Parameter(embedding)

  def forward(self, x):
    n, c, h, w = x.shape
    assert c == self.embedding_dim, "Input channels must equal embedding_dim."
    
    flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
    # Efficient L2 distance computation which does not require materializing the
    # huge NWH * n_embeddings * embedding_dim matrix. The computation follows
    # straightforwardly from Euclidian distance definition. For more info, see
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html.
    distances = (torch.sum(flat_x**2, dim=1, keepdim=True) + 
                 torch.sum(self._embedding**2, dim=1) - 
                 2 * flat_x @ self._embedding.t())

    # Quantize to closest embedding vector.
    idxs = torch.argmin(distances, dim=1, keepdim=True)
    one_hot = torch.zeros(idxs.shape[0], self.n_embeddings, 
                          device=self._embedding.device)
    one_hot.scatter_(1, idxs, 1)
    quantized = (one_hot @ self._embedding)
    quantized = quantized.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    # NOTE: Most implementations weight the commitment loss by some constant
    # given by the user. However, we find a weight of 1 is quite robust.
    loss = F.mse_loss(x, quantized.detach())  # Commitment loss.
    if self._use_ema and self.training:
      batch_cluster_size = one_hot.sum(axis=0)
      batch_embedding_avg = (flat_x.t() @ one_hot).t()
      self._cluster_size.data.mul_(self._decay).add_(batch_cluster_size, 
                                                     alpha=1 - self._decay)
      self._embedding_avg.data.mul_(self._decay).add_(batch_embedding_avg,
                                                      alpha=1 - self._decay)
      new_emb = self._embedding_avg / (self._cluster_size + 1e-5).unsqueeze(1)
      self._embedding.data.copy_(new_emb)
    elif not self._use_ema:
      # Add the embedding loss when not using EMA.
      loss += F.mse_loss(quantized, x.detach())

    quantized = x + (quantized - x).detach()  # Straight through estimator.
    return quantized, loss

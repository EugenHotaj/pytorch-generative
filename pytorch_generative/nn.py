"""Modules, funcitons, and  building blocks for Generative Neural Networks.

References (used throughout the code):
  [1]: https://arxiv.org/abs/1601.06759
  [2]: https://arxiv.org/abs/1712.09763
"""

import functools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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

  N.B.: This layer does *not* implement autoregressive channel masking.
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
def _get_causal_mask(size, is_causal=False):
  """Generates causal masks for attention weights."""
  return torch.tril(torch.ones((size, size)), diagonal=-int(is_causal))


# TODO(eugenhotaj): Do we need to expose an is_causal argument here?
class MaskedAttention(nn.Module):
  """Autoregresively masked multihead self-attention layer.

  Autoregressive masking means that the current pixel can only attend to itself,
  pixels to the left, and pixels above. 

  This Module generalizes attention to use 2D convolutions instead of fully 
  connected layers. As such, the input is expected to be 4D image tensors.
  """ 

  def __init__(self, 
               in_channels, 
               n_heads=1,
               embed_channels=None,
               out_channels=None,
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
    """
    super().__init__()
    self._n_heads = n_heads
    self._embed_channels = embed_channels or in_channels
    self._out_channels = out_channels or in_channels 

    self._query = nn.Conv2d(in_channels=in_channels, 
                            out_channels=self._embed_channels, kernel_size=1)
    self._kv = nn.Conv2d(in_channels=in_channels + extra_input_channels,
                         out_channels=embed_channels + out_channels, 
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
    n, _, h, w = x.shape 

    # Compute the q[uery], k[ey], and v[alue].
    q = self._query(x).view(n, self._embed_channels, -1)
    if extra_x is not None:
      x = torch.cat((x, extra_x), dim=1)
    kv = self._kv(x)
    k = kv[:, :self._embed_channels, :, :].view(n, self._embed_channels, -1)
    v = kv[:, self._embed_channels:, :, :].view(n, self._out_channels, -1)

    # Transpose q, k, v, to be in 'channels last' format as this is more common
    # in the literature and other libraries.
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # Compute the causual attention weights.
    mask = _get_causal_mask(h * w).to(next(self.parameters()).device)
    attn = (q @ k.transpose(1, 2)) / np.sqrt(self._embed_channels)
    attn = attn.masked_fill(mask == 0, -np.inf)
    attn = F.softmax(attn, dim=-1)

    return (attn @ v).transpose(1, 2).view(n, -1, h, w)

 

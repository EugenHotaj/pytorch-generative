"""Modules, funcitons, and  building blocks for Generative Neural Networks.

References (used throughout the code):
  [1]: https://arxiv.org/abs/1601.06759
  [2]: https://arxiv.org/abs/1712.09763
"""

import functools

import torch
from torch import nn


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
  return torch.tril(torch.ones((size, size), diagonal=-int(is_causal)))


class MaskedAttention(nn.Module):
  """A 2d Attention layer masked to respect the autoregressive property.

  Autoregressive masking means that the current pixel can only attend to itself,
  pixels to the left, and pixels above. When the attention weights are causally
  masked (i.e. is_causal=True), the computation of the current pixel does not
  depend on itself.
  """ 

  def __init__(self, 
               query_channels, 
               key_channels,
               value_channels,
               is_causal=False,
               kv_extra_channels=0):
    """Initializes a new MaskedAttention instance.

    Args:
      query_channels: Number of (input) query channels.
      key_channels: Number of key channels (i.e. key dimension).
      value_channels: Number of (output) value channels.
      is_causal: Whether the attention weights should be  causually masked. 
      kv_extra_channels: Extra channels to use as input to the key and
        value convolutions only. This is useful when using these channels as 
        query inputs would break the autoregressive property. For example, in 
        [2], the extra channels include the original input image.
    """
    super().__init__()
    self._key_channels = key_channels
    self._value_channels = value_channels

    kv_in_channels = query_channels + kv_extra_channels
    self._query = nn.Conv2d(
        in_channels=query_channels, 
        out_channels=self._key_channels, 
        kernel_size=1)
    self._kv = nn.Conv2d(
        in_channels=query_channels + kv_extra_channels, 
        out_channels=self._key_channels + self._value_channels, 
        kernel_size=1)
 
  def forward(self, x, kv_extra_channels=None):
    """Computes the forward pass.

    Args:
      x: The input used for the query, key, and value convolutions.
      kv_extra_channels: Extra channels concatenated with x which are only
        used as input to the key and value convolutions.
    Returns:
      The result of the forward pass.
    """
    n, _, h, w = x.shape 

    # Compute the query, key, and value.
    query = self._query(x).view(n, self._key_channels, -1)
    if kv_extra_channels is not None:
      x = torch.cat((x, kv_extra_channels), dim=1)
    kv = self._kv(x)
    key = kv[:, :self._key_channels, :, :].view(n, self._key_channels, -1)
    value = kv[:, self._key_channels:, :, :].view(n, self._value_channels, -1)

    # Compute the causual attention weights using stable softmax.
    mask = _get_causal_mask(h * w).to(next(self.parameters()).device)
    probs = (query.permute(0, 2, 1) @ key) - (1. - mask) * 1e10
    probs = probs - probs.max(dim=-1, keepdim=True)[0]
    probs = torch.exp(probs / np.sqrt(self._key_channels)) * mask
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)
    
    return (value @ probs.permute(0, 2, 1)).view(n, -1, h, w) 

 

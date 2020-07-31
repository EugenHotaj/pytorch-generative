"""Implementation of PixelCNN [1].

TODO(eugenhotaj): Explain.

NOTE: Our implementation does *not* use autoregressive channel masking. This
means that each output depends on whole pixels and not sub-pixels. For outputs
with multiple channels, other methods can be used, e.g. [2].

[1]: https://arxiv.org/abs/1606.05328
[2]: https://arxiv.org/abs/1701.05517
"""

import torch
from torch import distributions
from torch import nn


class MaskedConv2d(nn.Conv2d):
  """A Conv2d layer masked to respect the autoregressive property.

  Autoregressive masking means that the computation of the current pixel only
  depends on itself, pixels to the left, and pixels above. When the convolution
  is causally masked (i.e. 'is_causal=True'), the computation of the current 
  pixel does not depend on itself.

  E.g. for a 3x3 kernel, the following masks are generated for each channel:
                      [[1 1 1],                   [[1 1 1]
      is_causal=False  [1 1 0],    is_causal=True  [1 0 0]
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

    assert h % 2 == 1, 'kernel_size cannot be even'
    
    mask = torch.zeros((i, o, h, w))
    mask.data[:, :, :h//2, :] = 1
    mask.data[:, :, h//2, :w//2 + int(not is_causal)] = 1
    self.register_buffer('mask', mask)

  def forward(self, x):
    self.weight.data *= self.mask
    return super(MaskedConv2d, self).forward(x)
    
    
class MaskedResidualBlock(nn.Module):
  """A residual block masked to respect the autoregressive property."""
  
  def __init__(self, n_channels):
    """Initializes a new MaskedResidualBlock instance.

    Args:
      n_channels: The number of input (and output) channels.
    """
    super().__init__()
    self._net = nn.Sequential(
        # NOTE(eugenhotaj): The PixelCNN paper users ReLU->Conv2d since they do
        # not use a ReLU in the first layer. 
        nn.Conv2d(in_channels=n_channels, 
                  out_channels=n_channels//2, 
                  kernel_size=1),
        nn.ReLU(),
        MaskedConv2d(is_causal=False,
                     in_channels=n_channels//2,
                     out_channels=n_channels//2,
                     kernel_size=3,
                     padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=n_channels//2, 
                  out_channels=n_channels,
                  kernel_size=1),
        nn.ReLU())

  def forward(self, x):
    return x + self._net(x)


class PixelCNN(nn.Module):
  """The PixelCNN model."""

  def __init__(self, 
               in_channels, 
               out_dims=1,
               n_residual=15,
               residual_channels=128, 
               head_channels=32):
    """Initializes a new PixelCNN instance.
    
    Args:
      in_channels: The number of channels in the input image (typically either 
        1 or 3 for black and white or color images respectively).
      out_dims: The dimension of the output. Given input of the form 
        (N, C, H, W), the output from the model will be (N, out_dim, C, H, W).
      n_residual: The number of residual blocks.
      residual_channels: The number of channels to use in the residual layers.
      head_channels: The number of channels to use in the two 1x1 convolutional
        layers at the head of the network.
    """
    super().__init__()

    self._input = MaskedConv2d(is_causal=True,
                               in_channels=in_channels,
                               out_channels=2*residual_channels, 
                               kernel_size=7, 
                               padding=3)
    self._masked_layers = nn.ModuleList([
        MaskedResidualBlock(n_channels=2*residual_channels) 
        for _ in range(n_residual_blocks) 
    ])
    self._head = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(in_channels=2*residual_channels, 
                  out_channels=head_channels, 
                  kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=head_channels, 
                  out_channels=in_channels, 
                  kernel_size=1),
        nn.Sigmoid())

  def forward(self, x):
    x = self._input(x)
    skip = torch.zeros_like(x) + x
    for layer in self._masked_layers:
      x = layer(x)
      skip += x
    return self._head(skip)

  # TODO(eugenhotaj): We need to update the sampling code so it can handle 
  # outputs with dim > 1. One thing that's unclear: should the sample method
  # be part of the model?
  def sample(self):
    """Samples a new image.
    
    Args:
      conditioned_on: An (optional) image to condition samples on. Only 
        dimensions with values < 0 will be sampled. For example, if 
        conditioned_on[i] = -1, then output[i] will be sampled conditioned on
        dimensions j < i. If 'None', an unconditional sample will be generated.
    """
    with torch.no_grad():
      device = next(self.parameters()).device
      conditioned_on = (torch.ones((1, 1,  28, 28)) * - 1).to(device)

      for row in range(28):
        for column in range(28):
          for channel in range(1):
            out = self.forward(conditioned_on)[:, channel, row, column]
            out = distributions.Bernoulli(probs=out).sample()
            conditioned_on[:, channel, row, column] = torch.where(
                conditioned_on[:, channel, row, column] < 0,
                out, 
                conditioned_on[:, channel, row, column])
      return conditioned_on

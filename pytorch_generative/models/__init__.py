"""Models available in PyTorch Generative."""

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_generative.models.gated_pixel_cnn import GatedPixelCNN
from pytorch_generative.models.made import MADE
from pytorch_generative.models.nade import NADE


# TODO(eugenhotaj): Move MaskedConv2d into a layers module.
class _MaskedConv2d(nn.Conv2d):

  def __init__(self, is_causal, *args, **kwargs):
    super().__init__(*args, **kwargs)

    i, o, h, w = self.weight.shape

    assert h % 2 == 1, 'kernel_size cannot be even'
    
    mask = torch.zeros((i, o, h, w))
    mask.data[:, :, :h//2, :] = 1
    mask.data[:, :, h//2, :w//2 + int(not is_causal)] = 1
    self.register_buffer('mask', mask)

  def forward(self, x):
    self.weight.data *= self.mask
    return super().forward(x)


class TinyCNN(nn.Module):
  """A small network used for sanity checks."""

  def __init__(self, n_channels):
      super().__init__()
      self._conv = _MaskedConv2d(
          is_causal=True, in_channels=n_channels, out_channels=n_channels, 
          kernel_size=3, padding=1)

  def forward(self, x):
    return F.sigmoid(self._conv(x))


__all__ = ['GatedPixelCNN', 'MADE', 'NADE', 'TinyCNN']

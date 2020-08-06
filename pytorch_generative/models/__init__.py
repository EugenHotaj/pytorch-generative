"""Models available in PyTorch Generative."""

import torch
from torch import nn

from pytorch_generative.models.gated_pixel_cnn import GatedPixelCNN
from pytorch_generative.models.made import MADE
from pytorch_generative.models.nade import NADE
# TODO(eugenhotaj): Move MaskedConv2d into a layers module.
from pytorch_generative.models.pixel_cnn import MaskedConv2d
from pytorch_generative.models.pixel_cnn import PixelCNN


class TinyCNN(nn.Module):
  """A small network used for sanity checks."""

  def __init__(self, n_channels):
      super().__init__()
      self._conv = MaskedConv2d(
          is_causal=True, in_channels=n_channels, out_channels=n_channels, 
          kernel_size=3, padding=1)

  def forward(self, x):
    return torch.sigmoid(self._conv(x))


__all__ = ['GatedPixelCNN', 'MADE', 'NADE', 'PixelCNN', 'TinyCNN']

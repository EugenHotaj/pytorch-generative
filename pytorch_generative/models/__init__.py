"""Models available in PyTorch Generative."""

import torch
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base
from pytorch_generative.models.gated_pixel_cnn import GatedPixelCNN
from pytorch_generative.models.made import MADE
from pytorch_generative.models.nade import NADE
from pytorch_generative.models.pixel_cnn import PixelCNN


class TinyCNN(base.AutoregressiveModel):
  """A small network used for sanity checks."""

  def __init__(self, in_channels):
      super().__init__()
      self._conv = pg_nn.MaskedConv2d(
          is_causal=True, in_channels=in_channels, out_channels=in_channels, 
          kernel_size=3, padding=1)

  def forward(self, x):
    return torch.sigmoid(self._conv(x))


__all__ = ['GatedPixelCNN', 'MADE', 'NADE', 'PixelCNN', 'TinyCNN']

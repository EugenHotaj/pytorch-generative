"""Models available in PyTorch Generative."""

import torch
from torch import distributions
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base
from pytorch_generative.models.gated_pixel_cnn import GatedPixelCNN
from pytorch_generative.models.image_gpt import ImageGPT
from pytorch_generative.models.made import MADE
from pytorch_generative.models.nade import NADE
from pytorch_generative.models.pixel_cnn import PixelCNN
from pytorch_generative.models.pixel_snail import PixelSNAIL
from pytorch_generative.models.vq_vae import VQVAE
from pytorch_generative.models.vq_vae import VQVAE2


class TinyCNN(base.AutoregressiveModel):
  """A small network used for sanity checks."""

  def __init__(self, 
      in_channels=1, 
      out_channels=1,
      sample_fn=None):
    """Initializes a new TinyCNN instance.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      sample_fn: See the base class.
    """
    super().__init__(sample_fn)
    self._conv = pg_nn.MaskedConv2d(
        is_causal=True, 
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3, 
        padding=1)

  def forward(self, x):
    return self._conv(x)


__all__ = [
    'GatedPixelCNN', 
    'ImageGPT',
    'MADE', 
    'NADE',
    'PixelCNN', 
    'PixelSNAIL', 
    'VQVAE',
    'VQVAE2',
    'TinyCNN'
]

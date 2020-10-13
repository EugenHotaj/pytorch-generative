"""Implementation of the Vector Quantized Variational Autoencoder (VQ-VAE) [1].

TODO(eugenhotaj): explain.

[1]: https://arxiv.org/pdf/1711.00937.pdf
"""

import torch
from torch import nn

from pytorch_generative import nn as pg_nn


class ResidualBlock(nn.Module):
  """A simple residual block."""

  def __init__(self, n_channels, hidden_channels=32):
    """Initializes a new ResidualBlock instance.
    
    Args:
      n_channels: Number of input and output channels.
      hidden_channels: Number of hidden channels.
    """
    super().__init__()
    self._net = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(in_channels=n_channels, out_channels=hidden_channels, 
                  kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels, out_channels=n_channels, 
                  kernel_size=1))

  def forward(self, x):
    return x + self._net(x)


class ResidualStack(nn.Module):
  """A stack of multiple ResidualBlocks."""

  def __init__(self, n_channels, hidden_channels=32, n_residual_blocks=2):
    """Initializes a new ResidualStack instance.
    
    Args:
      n_channels: Number of input and output channels.
      hidden_channels: Number of hidden channels.
      n_residual_blocks: Number of residual blocks in the stack.
    """
    super().__init__()
    self._net = nn.Sequential(*[
        ResidualBlock(n_channels, hidden_channels)
        for _ in range(n_residual_blocks)
    ])

  def forward(self, x):
    return torch.relu(self._net(x))


class VQVAE(nn.Module):
  """The Vector Quantized Variational Autoencoder (VQ-VAE) model."""

  def __init__(self,
               in_channels=1,
               out_channels=1,
               hidden_channels=128,
               residual_hidden_channels=32,
               n_residual_blocks=2,
               n_embeddings=128,
               embedding_dim=16):
    """Initializes a new VQVAE instance.
    
    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      hidden_channels: Number of non-ResidualBlock hidden channels.
      residual_hidden_channels: Number of ResidualBlock hidden channels.
      n_residual_blocks: Number of ResidualBlocks in encoder/decoder stacks.
      n_embeddings: Number of vectors to use in the quantizaiton dictionary. 
      embedding_dim: Dimension of each quantization vector.
    """
    super().__init__()
    self._encoder = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels//2,
                  kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels//2, out_channels=hidden_channels,
                  kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, 
                  kernel_size=3, padding=1),
        ResidualStack(n_channels=hidden_channels,
                      hidden_channels=residual_hidden_channels,
                      n_residual_blocks=n_residual_blocks))
    self._quantizer = nn.Sequential(
        nn.Conv2d(in_channels=hidden_channels, out_channels=embedding_dim,
                  kernel_size=1),
        pg_nn.VectorQuantizer(n_embeddings, embedding_dim))
    self._decoder = nn.Sequential(
        nn.Conv2d(in_channels=embedding_dim, out_channels=hidden_channels,
                kernel_size=3, padding=1),
        ResidualStack(n_channels=hidden_channels,
                      hidden_channels=residual_hidden_channels,
                      n_residual_blocks=n_residual_blocks),
        nn.ConvTranspose2d(in_channels=hidden_channels, 
                           out_channels=hidden_channels//2, kernel_size=4,
                           stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=hidden_channels//2, 
                           out_channels=out_channels, kernel_size=4,
                           stride=2, padding=1))

  def forward(self, x):
    x = self._encoder(x)
    quantized, quantization_loss = self._quantizer(x)
    return self._decoder(quantized), quantization_loss

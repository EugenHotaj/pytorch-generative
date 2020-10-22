"""Implementation of the VQ-VAE [1] and VQ-VAE-2 [2] models.

The Vector Quantized Variational Autoencoder (VQ-VAE) extends the vanilla 
Variational Autoencoder by mapping the input to a discrete (instead of 
continuous) latent space. The discretization is performed using vector 
quantization where continuous outputs from an encoder are mapped to the nearest
vectors in a learned codebook. The main intuition behind using discretized
latents is that the underlying domain we are trying to model is usually 
discrete (e.g. words in a sentence, objects in an image, etc).

VQ-VAE-2 further extends the original VQ-VAE by encoding the input to a 
hierarchy of latent spaces. The top latent space is responsible for encoding 
high level semantic details with progressively lower level details being encoded
as down the hierarchy.

References (used throughout the code):
  [1]: https://arxiv.org/pdf/1711.00937.pdf
  [2]: https://arxiv.org/pdf/1906.00446.pdf
"""

import torch
from torch import nn

from pytorch_generative import nn as pg_nn


class ResidualBlock(nn.Module):
  """A simple residual block."""

  def __init__(self, n_channels, hidden_channels):
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

  def __init__(self, n_channels, hidden_channels, n_residual_blocks=1):
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
    ] + [nn.ReLU()])

  def forward(self, x):
    return self._net(x)


class Encoder(nn.Module):
  """A feedforward encoder which downsamples its input."""

  def __init__(self,
               in_channels, 
               out_channels, 
               hidden_channels,
               n_residual_blocks,
               residual_channels, 
               stride):
    """Initializes a new Encoder instance.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      hidden_channels: Number of channels in non residual block hidden layers.
      n_residual_blocks: Number of residual blocks in each residual stack. 
      residual_channels: Number of hidden channels in residual blocks.
      stride: Stride to use in the downsampling convolutions. Must be even.
    """
    super().__init__()
    assert stride % 2 == 0, '"stride" must be even.'

    net = []
    for i in range(stride//2):
      first, last = 0, stride // 2 - 1
      in_c = in_channels if i == first else hidden_channels//2
      out_c = hidden_channels //2 if i < last else hidden_channels 
      net.append(
          nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, 
                    stride=2, padding=1))
      net.append(nn.ReLU())
    net.append(
        nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                  kernel_size=3, padding=1))
    net.append(
        ResidualStack(n_channels=out_channels,
                      hidden_channels=residual_channels,
                      n_residual_blocks=n_residual_blocks))
    self._net = nn.Sequential(*net)

  def forward(self, x):
    return self._net(x)


class Decoder(nn.Module):
  """A feedforward encoder which upsamples its input."""

  def __init__(self,
               in_channels,
               out_channels,
               hidden_channels,
               n_residual_blocks,
               residual_channels,
               stride):
    """Initializes a new Decoder instance.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      hidden_channels: Number of channels in (non residual block) hidden layers.
      n_residual_blocks: Number of residual blocks in each residual stack. 
      residual_channels: Number of hidden channels in residual blocks.
      stride: Stride to use in the upsampling (i.e. transpose) convolutions. 
        Must be even.
    """
    super().__init__()

    assert stride % 2 == 0, '"stride" must be even.'

    net = [
      nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels,
                kernel_size=3, padding=1),
      ResidualStack(n_channels=hidden_channels,
                    hidden_channels=residual_channels,
                    n_residual_blocks=n_residual_blocks)
    ]
    for i in range(stride//2):
      first, last = 0, stride//2 - 1
      in_c = hidden_channels if i == first else hidden_channels//2
      out_c = hidden_channels//2 if i < last else out_channels
      net.append(
          nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c,
                             kernel_size=4, stride=2, padding=1))
      if i < last:
        net.append(nn.ReLU()) 
    self._net = nn.Sequential(*net)

  def forward(self, x):
    return self._net(x)


class Quantizer(nn.Module):
  """Wraps a VectorQuantizer to handle input with arbitrary channels."""

  def __init__(self, in_channels, n_embeddings, embedding_dim):
    """Initializes a new Quantizer instance.

    Args:
      in_channels: Number of input channels.
      n_embeddings: Number of VectorQuantizer embeddings.
      embedding_dim: VectorQuantizer embedding dimension.
    """
    super().__init__()
    self._net = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                  kernel_size=1),
        pg_nn.VectorQuantizer(n_embeddings, embedding_dim))

  def forward(self, x):
    return self._net(x)


class VQVAE(nn.Module):
  """The Vector Quantized Variational Autoencoder (VQ-VAE) model."""

  def __init__(self,
               in_channels=1,
               out_channels=1,
               hidden_channels=128,
               n_residual_blocks=2,
               residual_channels=32,
               n_embeddings=128,
               embedding_dim=16):
    """Initializes a new VQVAE instance.
    
    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      hidden_channels: Number of channels in (non residual block) hidden layers.
      n_residual_blocks: Number of residual blocks in each residual stack.
      residual_channels: Number of hidden channels in residual blocks.
      n_embeddings: Number of VectorQuantizer embeddings.
      embedding_dim: Dimension of the VectorQuantizer embeddings.
    """
    super().__init__()
    self._encoder = Encoder(in_channels=in_channels,
                            out_channels=hidden_channels, 
                            hidden_channels=hidden_channels,
                            n_residual_blocks=n_residual_blocks,
                            residual_channels=residual_channels, 
                            stride=4)
    self._quantizer = Quantizer(in_channels=hidden_channels,
                                n_embeddings=n_embeddings, 
                                embedding_dim=embedding_dim)
    self._decoder = Decoder(in_channels=embedding_dim,
                            out_channels=out_channels,
                            hidden_channels=hidden_channels,
                            n_residual_blocks=n_residual_blocks,
                            residual_channels=residual_channels,
                            stride=4) 

  def forward(self, x):
    x = self._encoder(x)
    quantized, quantization_loss = self._quantizer(x)
    return self._decoder(quantized), quantization_loss
    

class VQVAE2(nn.Module):
  """The VQ-VAE-2 model with a latent hierarchy of depth 2."""

  def __init__(self,
               in_channels=1,
               out_channels=1,
               hidden_channels=128,
               n_residual_blocks=2,
               residual_channels=32,
               n_embeddings=128,
               embedding_dim=16):
    """Initializes a new VQVAE2 instance.
    
    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      hidden_channels: Number of channels in (non residual block) hidden layers.
      n_residual_blocks: Number of residual blocks in each residual stack.
      residual_channels: Number of hidden channels in residual blocks.
      n_embeddings: Number of VectorQuantizer embeddings.
      embedding_dim: Dimension of the VectorQuantizer embeddings.
    """
    super().__init__()

    self._encoder_b = Encoder(in_channels=in_channels, 
                              out_channels=hidden_channels, 
                              hidden_channels=hidden_channels,
                              n_residual_blocks=n_residual_blocks,
                              residual_channels=residual_channels, 
                              stride=2)
    self._encoder_t = Encoder(in_channels=hidden_channels,
                              out_channels=hidden_channels, 
                              hidden_channels=hidden_channels,
                              n_residual_blocks=n_residual_blocks,
                              residual_channels=residual_channels,
                              stride=2)
    self._quantizer_t = Quantizer(in_channels=hidden_channels, 
                                  n_embeddings=n_embeddings, 
                                  embedding_dim=embedding_dim)
    self._quantizer_b = Quantizer(in_channels=hidden_channels, 
                                  n_embeddings=n_embeddings, 
                                  embedding_dim=embedding_dim)
    self._decoder_t = Decoder(in_channels=embedding_dim,
                              out_channels=hidden_channels,
                              hidden_channels=hidden_channels,
                              n_residual_blocks=n_residual_blocks,
                              residual_channels=residual_channels,
                              stride=2)
    self._conv = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                          kernel_size=1)
    self._decoder_b = Decoder(in_channels=2*embedding_dim,
                              out_channels=out_channels,
                              hidden_channels=hidden_channels,
                              n_residual_blocks=n_residual_blocks,
                              residual_channels=residual_channels,
                              stride=2) 
    
  def forward(self, x):
    encoded_b = self._encoder_b(x)
    encoded_t = self._encoder_t(encoded_b)

    quantized_t, vq_loss_t = self._quantizer_t(encoded_t)
    quantized_b, vq_loss_b = self._quantizer_b(encoded_b)

    decoded_t = self._decoder_t(quantized_t)
    xhat = self._decoder_b(
        torch.cat((self._conv(decoded_t), quantized_b), dim=1))
    return xhat, .5 * (vq_loss_b + vq_loss_t) + F.mse_loss(decoded_t, encoded_b)


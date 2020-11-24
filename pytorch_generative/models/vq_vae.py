"""Implementation of the VQ-VAE [1] model.

The Vector Quantized Variational Autoencoder (VQ-VAE) extends the vanilla 
Variational Autoencoder by mapping the input to a discrete (instead of 
continuous) latent space. The discretization is performed using vector 
quantization where continuous outputs from an encoder are mapped to the nearest
vectors in a learned codebook. The main intuition behind using discretized
latents is that the underlying domain we are trying to model is usually 
discrete (e.g. words in a sentence, objects in an image, etc).

References (used throughout the code):
  [1]: https://arxiv.org/pdf/1711.00937.pdf
"""

from torch import nn

from pytorch_generative.models import vaes


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
    self._encoder = vaes.Encoder(in_channels=in_channels,
                                 out_channels=hidden_channels, 
                                 hidden_channels=hidden_channels,
                                 n_residual_blocks=n_residual_blocks,
                                 residual_channels=residual_channels, 
                                 stride=4)
    self._quantizer = vaes.Quantizer(in_channels=hidden_channels,
                                     n_embeddings=n_embeddings, 
                                     embedding_dim=embedding_dim)
    self._decoder = vaes.Decoder(in_channels=embedding_dim,
                                 out_channels=out_channels,
                                 hidden_channels=hidden_channels,
                                 n_residual_blocks=n_residual_blocks,
                                 residual_channels=residual_channels,
                                 stride=4) 

  def forward(self, x):
    x = self._encoder(x)
    quantized, quantization_loss = self._quantizer(x)
    return self._decoder(quantized), quantization_loss
    

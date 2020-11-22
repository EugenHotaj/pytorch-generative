"""TODO(eugenhotaj): explain."""

import torch
from torch import nn

from pytorch_generative.models import vaes


class VAE(nn.Module):
  """The Variational Autoencoder Model."""

  def __init__(self,
               in_channels=3,
               out_channels=3,
               in_size=32,
               latent_dim=16,
               hidden_channels=128,
               n_residual_blocks=2,
               residual_channels=64):
    """Initializes a new VAE instance.
    
    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      in_size: Size of the input images. Used to create bottleneck layers.
      latent_dim: The dimensionality of each latent variable.
      hidden_channels: Number of channels in (non residual block) hidden layers.
      n_residual_blocks: Number of residual blocks in each residual stack.
      residual_channels: Number of hidden channels in residual blocks.
    """
    super().__init__()

    self._latent_dim = latent_dim
    self._encoder_out_dims = (hidden_channels, in_size // 4, in_size // 4)

    self._encoder = vaes.Encoder(in_channels=in_channels,
                                 out_channels=hidden_channels,
                                 hidden_channels=hidden_channels,
                                 residual_channels=residual_channels, 
                                 n_residual_blocks=n_residual_blocks,
                                 stride=4)
    flat_dim = hidden_channels * in_size // 4 * in_size // 4
    self._mean = nn.Linear(flat_dim, self._latent_dim)
    self._log_var = nn.Linear(flat_dim, self._latent_dim)
    self._bottleneck = nn.Linear(self._latent_dim, flat_dim) 
    self._decoder_ = vaes.Decoder(in_channels=hidden_channels,
                                  out_channels=out_channels,
                                  hidden_channels=hidden_channels,
                                  residual_channels=residual_channels,
                                  n_residual_blocks=n_residual_blocks,
                                  stride=4)

  def _decoder(self, x):
    n, c, h, w = x.shape[0], *self._encoder_out_dims
    x  = self._bottleneck(x).view(n, c, h, w)
    return self._decoder_(x)

  def forward(self, x):
    encoded = self._encoder(x).view(x.shape[0], -1)
    # NOTE: We use log_var (instead of var or std) for stability and easier
    # optimization during training. 
    mean, log_var = self._mean(encoded), self._log_var(encoded)
    var = torch.exp(log_var)
    latents = mean + torch.sqrt(var) * torch.randn_like(var)
    # NOTE: This KL divergence is only applicable under the assumption that the
    # prior ~ N(0, 1) and the latents are Gaussian.
    # NOTE: Technically, the KL divergence computation requires a sum over the 
    # latent dimension. However, this makes the magnitude of the KL divergence 
    # dependent on the dimensionality of the latents. Instead, we divide the
    # KL divergence by the dimension of the latents, which is equivalent to 
    # averaging over the latent dimensions.
    kl_div = -.5 * (1 + log_var - mean ** 2 - var).mean()
    return self._decoder(latents), kl_div

  def sample(self, n_images):
    """Generates a batch of n_images."""
    device = next(self.parameters()).device
    latents = torch.randn((n_images, self._latent_dim), device=device)
    return self._decoder(latents)



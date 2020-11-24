"""TODO(eugenhotaj): explain."""

import torch
from torch import nn

from pytorch_generative.models import vaes


class VAE(nn.Module):
  """The Variational Autoencoder Model."""

  def __init__(self,
               in_channels=1,
               out_channels=1,
               in_size=28,
               latent_dim=10,
               hidden_channels=32,
               n_residual_blocks=2,
               residual_channels=16):
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


def reproduce(n_epochs=457, batch_size=128, log_dir='/tmp/run', device='cuda', 
              debug_loader=None):
  """Training script with defaults to reproduce results.

  The code inside this function is self contained and can be used as a top level
  training script, e.g. by copy/pasting it into a Jupyter notebook.

  Args:
    n_epochs: Number of epochs to train for.
    batch_size: Batch size to use for training and evaluation.
    log_dir: Directory where to log trainer state and TensorBoard summaries.
    device: Device to train on (either 'cuda' or 'cpu').
    debug_loader: Debug DataLoader which replaces the default training and 
      evaluation loaders if not 'None'. Do not use unless you're writing unit
      tests.
  """
  from torch import optim
  from torch.nn import functional as F
  from torch.optim import lr_scheduler
  from torch.utils import data
  from torchvision import datasets
  from torchvision import transforms

  from pytorch_generative import trainer
  from pytorch_generative import models

  transform = transforms.ToTensor()
  train_loader = debug_loader or data.DataLoader(
      datasets.MNIST('/tmp/data', train=True, download=True, transform=transform),
      batch_size=batch_size, 
      shuffle=True,
      num_workers=8)
  test_loader = debug_loader or data.DataLoader(
      datasets.MNIST('/tmp/data', train=False, download=True,transform=transform),
      batch_size=batch_size,
      num_workers=8)

  model = models.VAE(in_channels=1,
                     out_channels=1,
                     in_size=28,
                     latent_dim=10,
                     hidden_channels=32,
                     n_residual_blocks=2,
                     residual_channels=16)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  scheduler = lr_scheduler.MultiplicativeLR(optimizer, 
                                            lr_lambda=lambda _: .999977)
  def loss_fn(x, _, preds):
    preds, vae_loss = preds 
    recon_loss = F.binary_cross_entropy_with_logits(preds, x)
    loss = recon_loss * 100 + vae_loss
    return {
        "recon_loss": recon_loss,
        "vae_loss": vae_loss,
        "loss": loss,
    }

  def sample_fn(model):
    return torch.sigmoid(model.sample(n_images=64))

  model_trainer = trainer.Trainer(model=model,
                                  loss_fn=loss_fn,
                                  optimizer=optimizer,
                                  train_loader=train_loader,
                                  eval_loader=test_loader,
                                  lr_scheduler=scheduler,
                                  sample_epochs=5,
                                  sample_fn=sample_fn,
                                  log_dir=log_dir,
                                  device=device)
  model_trainer.interleaved_train_and_eval(n_epochs)

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

  transform = transforms.Compose([
      transforms.ToTensor(), 
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  train_loader = data.DataLoader(
      datasets.CIFAR10('tmp/data', train=True, download=True, transform=transform),
      batch_size=batch_size, 
      shuffle=True,
      pin_memory=True,
      num_workers=2)
  test_loader = data.DataLoader(
      datasets.CIFAR10('tmp/data', train=False, download=True,transform=transform),
      batch_size=batch_size,
      pin_memory=True,
      num_workers=2)

  model = models.VQVAE(in_channels=3,
                       out_channels=3,
                       hidden_channels=128,
                       residual_channels=32, 
                       n_residual_blocks=2,
                       n_embeddings=512, 
                       embedding_dim=64)
  optimizer = optim.Adam(model.parameters(), lr=2e-4)
  scheduler = lr_scheduler.MultiplicativeLR(optimizer, 
                                            lr_lambda=lambda _: .999977)

  def loss_fn(x, _, preds):
    preds, vq_loss = preds
    recon_loss = F.mse_loss(preds, x)
    loss = recon_loss + vq_loss 

    return {
        'vq_loss': vq_loss,
        'reconstruction_loss': recon_loss,
        'loss': loss, 
    }

  model_trainer = trainer.Trainer(model=model, 
                                  loss_fn=loss_fn, 
                                  optimizer=optimizer, 
                                  train_loader=train_loader, 
                                  eval_loader=test_loader,
                                  lr_scheduler=scheduler,
                                  log_dir=log_dir,
                                  device=device)
  model_trainer.interleaved_train_and_eval(n_epochs)

"""Implementation of Neural Autoregressive Distribution Estimator (NADE) [1].

NADE can be viewed as a one hidden layer autoencoder masked to satisfy the 
autoregressive property. This masking allows NADE to act as a generative model
by explicitly estimating p(X) as a factor of conditional probabilities, i.e,
P(X) = \prod_i^D p(X_i|X_{j<i}), where X is a feature vector and D is the 
dimensionality of X.

[1]: https://arxiv.org/abs/1605.02226
"""

import torch
from torch import distributions
from torch import nn

from pytorch_generative.models import base


class NADE(base.AutoregressiveModel):
  """The Neural Autoregressive Distribution Estimator (NADE) model."""

  def __init__(self, input_dim, hidden_dim):
    """Initializes a new NADE instance.

    Args:
      input_dim: The dimension of the input.
      hidden_dim: The dimmension of the hidden layer. NADE only supports one
        hidden layer.
    """
    super().__init__()
    self._input_dim = input_dim
    self._hidden_dim = hidden_dim
    self.params = nn.ParameterDict({
      'in_W': nn.Parameter(torch.zeros(self._hidden_dim, self._input_dim)),
      'in_b': nn.Parameter(torch.zeros(self._hidden_dim,)),
      'h_W': nn.Parameter(torch.zeros(self._input_dim, self._hidden_dim)),
      'h_b': nn.Parameter(torch.zeros(self._input_dim,)),
    })
    nn.init.kaiming_normal_(self.params['in_W'])
    nn.init.kaiming_normal_(self.params['h_W'])

  def _forward(self, x):
    """Computes the forward pass and samples a new output.
    
    Returns:
      (p_hat, x_hat) where p_hat is the probability distribution over dimensions
      and x_hat is sampled from p_hat.
    """
    # If the input is an image, flatten it during the forward pass.
    original_shape = x.shape
    if len(x.shape) > 2:
      x = x.view(original_shape[0], -1)

    in_W, in_b = self.params['in_W'], self.params['in_b']
    h_W, h_b = self.params['h_W'], self.params['h_b']
    batch_size = 1 if x is None else x.shape[0] 

    p_hat = []
    x_hat = []
    # Only the bias is used to compute the first hidden unit so we must 
    # replicate it to account for the batch size.
    a = in_b.expand(batch_size, -1)
    for i in range(self._input_dim):
      h = torch.relu(a)
      p_i = torch.sigmoid(h_b[i:i+1] + h @ h_W[i:i+1, :].t())
      p_hat.append(p_i)

      # Sample 'x' at dimension 'i' if it is not given.
      x_i = x[:, i:i+1]
      x_i = torch.where(x_i < 0, 
                        distributions.Bernoulli(probs=p_i).sample(),
                        x_i)
      x_hat.append(x_i)

      # We do not need to add in_b[i:i+1] when computing the other hidden units
      # since it was already added when computing the first hidden unit. 
      a = a + x_i @ in_W[:, i:i+1].t()
    if x_hat:
      return (torch.cat(p_hat, dim=1).view(original_shape), 
              torch.cat(x_hat, dim=1).view(original_shape))
    return []

  def forward(self, x):
    """Computes the forward pass.

    Args:
      x: Either a tensor of vectors with shape (n, input_dim) or images with
        shape (n, 1, h, w) where h * w = input_dim.
    Returns:
      The result of the forward pass.
    """
    return self._forward(x)[0]

  # TODO(eugenhotaj): It's kind of dumb to require an out_shape for 
  # non-convolutional models. We already know what the out_shape should be based
  # on the model parameters.orks
  def sample(self, out_shape=None, conditioned_on=None):
    """See the base class."""
    with torch.no_grad():
      conditioned_on = self._get_conditioned_on(out_shape, conditioned_on)
      return self._forward(conditioned_on)[1]


def reproduce(n_epochs=50, batch_size=512, log_dir='/tmp/run', device='cuda', 
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
  from torch import distributions
  from torch.nn import functional as F
  from torch.optim import lr_scheduler
  from torch.utils import data
  from torchvision import datasets
  from torchvision import transforms

  from pytorch_generative import trainer
  from pytorch_generative import models

  transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: distributions.Bernoulli(probs=x).sample()])
  train_loader = debug_loader or data.DataLoader(
      datasets.MNIST('/tmp/data', train=True, download=True, transform=transform),
      batch_size=batch_size, 
      shuffle=True,
      num_workers=8)
  test_loader = debug_loader or data.DataLoader(
      datasets.MNIST('/tmp/data', train=False, download=True,transform=transform),
      batch_size=batch_size,
      num_workers=8)

  model = models.NADE(input_dim=784, hidden_dim=500)
  optimizer = optim.Adam(model.parameters())

  def loss_fn(x, _, preds):
    batch_size = x.shape[0] 
    x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
    loss = F.binary_cross_entropy_with_logits(preds, x, reduction='none')
    return loss.sum(dim=1).mean()

  model_trainer = trainer.Trainer(model=model,
                                  loss_fn=loss_fn, 
                                  optimizer=optimizer, 
                                  train_loader=train_loader, 
                                  eval_loader=test_loader,
                                  log_dir=log_dir,
                                  device=device)
  model_trainer.interleaved_train_and_eval(n_epochs)


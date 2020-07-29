"""Implementation of Neural Autoregressive Distribution Estimator (NADE) [1].

NADE can be viewed as a one hidden layer autoencoder which is masked to satisfy
the autoregressive property. This allows NADE to act as a generative model by
explicitly estimating p(X) as a factor of conditional probabilities, i.e, P(X) =
\prod_i^D p(X_i|X_{j<i}), where X is a feature vector and D is the 
dimensionality of X. For the full details, see [1].

[1]: https://arxiv.org/abs/1605.02226
"""

import torch
from torch import nn


class NADE(nn.Module):
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
    return torch.cat(p_hat, dim=1), torch.cat(x_hat, dim=1) if x_hat else []

  def forward(self, x):
    """Computes the forward pass."""
    return self._forward(x)[0]

  def sample(self, conditioned_on=None):
    """Samples a new image.
    
    Args:
      conditioned_on: An (optional) image to condition samples on. Only 
        dimensions with values < 0 will be sampled. For example, if 
        conditioned_on[i] = -1, then output[i] will be sampled conditioned on
        dimensions j < i. If 'None', an unconditional sample will be generated.
    """
    with torch.no_grad():
      if conditioned_on is None:
        device = next(self.parameters()).device
        conditioned_on = (torch.ones((1, self._input_dim)) * -1).to(device)
      return self._forward(conditioned_on)[1]

"""Implementation of Masked Autoencoder Distribution Estimator (MADE) [1].

MADE is an extension of NADE [2] which allows using arbitrarily deep fully 
connected networks as the distribution estimator. More specifically, MADE is a
deep, fully-connected autoencoder masked to respect the autoregressive property.
For any ordering of the input features, MADE only uses features j<i to predict 
feature i. This property allows MADE to be used as a generative model by 
specifically modelling P(X) = \prod_i^D p(X_i|X_{j<i}) where X is an input
feature and D is the dimensionality of X.

[1]: https://arxiv.org/abs/1502.03509
[2]: https://arxiv.org/abs/1605.02226
"""

import numpy as np
import torch
from torch import distributions
from torch import nn

from pytorch_generative.models import base


class MaskedLinear(nn.Linear):
  """A Linear layer with masks that turn off some of the layer's weights."""

  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias)
    self.register_buffer('mask', torch.ones((out_features, in_features)))

  def set_mask(self, mask):
    self.mask.data.copy_(mask)

  def forward(self, x):
    self.weight.data *= self.mask
    return super().forward(x)


class MADE(base.AutoregressiveModel):
  """The Masked Autoencoder Distribution Estimator (MADE) model."""

  def __init__(self, input_dim, hidden_dims=None, n_masks=1):
    """Initializes a new MADE instance.
    
    Args:
      input_dim: The dimensionality of the input.
      hidden_dims: A list containing the number of units for each hidden layer.
      n_masks: The total number of distinct masks to use during training/eval.
    """
    super().__init__()
    self._input_dim = input_dim
    self._dims = [self._input_dim] + (hidden_dims or []) + [self._input_dim]
    self._n_masks = n_masks
    self._mask_seed = 0

    layers = []
    for i in range(len(self._dims) - 1):
      in_dim, out_dim = self._dims[i], self._dims[i + 1]
      layers.append(MaskedLinear(in_dim, out_dim))
      layers.append(nn.ReLU())
    layers[-1] = nn.Sigmoid()  # Output is binary. 
    self._net = nn.Sequential(*layers)

  def _sample_masks(self):
    """Samples a new set of autoregressive masks.

    Only 'self._n_masks' distinct sets of masks are sampled after which the mask
    sets are rotated through in the order in which they were sampled. In 
    principle, it's possible to generate the masks once and cache them. However,
    this can lead to memory issues for large 'self._n_masks' or models many
    parameters. Finally, sampling the masks is not that computationally 
    expensive. 

    Returns:
      A tuple of (masks, ordering). Ordering refers to the ordering of the 
        outputs since MADE is order agnostic.
    """
    rng = np.random.RandomState(seed=self._mask_seed % self._n_masks)
    self._mask_seed += 1

    # Sample connectivity patterns.
    conn = [rng.permutation(self._input_dim)]
    for i, dim in enumerate(self._dims[1:-1]):
      # NOTE(eugenhotaj): The dimensions in the paper are 1-indexed whereas 
      # arrays in Python are 0-indexed. Implementation adjusted accordingly. 
      low = 0 if i == 0 else np.min(conn[i - 1])
      high = self._input_dim - 1
      conn.append(rng.randint(low, high, size=dim))
    conn.append(np.copy(conn[0]))

    # Create masks.
    masks = [conn[i - 1][None, :] <= conn[i][:, None] 
             for i in range(1, len(conn) - 1 )]
    masks.append(conn[-2][None, :] < conn[-1][:, None])

    return [torch.from_numpy(mask.astype(np.uint8)) for mask in masks], conn[-1]

  def _forward(self, x, masks): 
    # If the input is an image, flatten it during the forward pass.
    original_shape = x.shape
    if len(original_shape) > 2:
      x = x.view(original_shape[0], -1)

    layers = [
      layer for layer in self._net.modules() if isinstance(layer, MaskedLinear)]
    for layer, mask in zip(layers, masks):
      layer.set_mask(mask)
    return self._net(x).view(original_shape)

  def forward(self, x):
    """Computes the forward pass.

    Args:
      x: Either a tensor of vectors with shape (n, input_dim) or images with
        shape (n, 1, h, w) where h * w = input_dim.
    Returns:
      The result of the forward pass.
    """

    masks, _ = self._sample_masks()
    return self._forward(x, masks)

  # TODO(eugenhotaj): It's kind of dumb to require an out_shape for 
  # non-convolutional models. We already know what the out_shape should be based
  # on the model parameters.
  def sample(self, out_shape=None, conditioned_on=None):
    """See the base class."""
    with torch.no_grad():
      conditioned_on = self._get_conditioned_on(out_shape, conditioned_on)
      out_shape = conditioned_on.shape
      conditioned_on = conditioned_on.view(out_shape[0], -1)

      masks, ordering = self._sample_masks()
      ordering = np.argsort(ordering)
      for dim in ordering:
        out = self._forward(conditioned_on, masks)[:, dim]
        out = distributions.Bernoulli(probs=out).sample()
        conditioned_on[:, dim] = torch.where(
            conditioned_on[:, dim] < 0, out, conditioned_on[:, dim])
      return conditioned_on.view(out_shape)

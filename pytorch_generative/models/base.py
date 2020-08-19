"""Base classes for models."""

import torch
from torch import distributions
from torch import nn


class AutoregressiveModel(nn.Module):
  """The base class for Autoregressive generative models. """

  def _get_conditioned_on(self, out_shape, conditioned_on):
    assert out_shape is None or conditioned_on is None, \
      'Must provided one, and only one of "out_shape" or "conditioned_on"'
    if conditioned_on is None:
      device = next(self.parameters()).device
      conditioned_on = (torch.ones(out_shape) * - 1).to(device)
    else:
      conditioned_on = conditioned_on.clone()
    return conditioned_on

  # TODO(eugenhotaj): It may be possible to support more complex output 
  # distributions by allowing the user to specify a sampling_fn.
  # TODO(eugenhotaj): This function does not handle subpixel sampling correctly.
  def sample(self, out_shape=None, conditioned_on=None):
    """Generates new samples from the model.

    The model output is assumed to be the parameters of either a Bernoulli or 
    multinoulli (Categorical) distribution depending on its dimension.

    Args:
      out_shape: The expected shape of the sampled output in NCHW format. 
        Should only be provided when 'conditioned_on=None'.
      conditioned_on: A batch of partial samples to condition the generation on.
        Only dimensions with values < 0 will be sampled while dimensions with 
        values >= 0 will be left unchanged. If 'None', an unconditional sample
        will be generated.
    """
    with torch.no_grad():
      conditioned_on = self._get_conditioned_on(out_shape, conditioned_on)
      n, c, h, w = conditioned_on.shape
      for row in range(h):
        for column in range(w):
          out = self.forward(conditioned_on)[:, :, :, row, column]
          distribution = (distributions.Categorical if out.shape[1] > 1 
                          else distributions.Bernoulli)
          out = distribution(probs=out).sample().view(n, c)
          conditioned_on[:, :, row, column] = torch.where(
              conditioned_on[:, :, row, column] < 0,
              out, 
              conditioned_on[:, :, row, column])
      return conditioned_on

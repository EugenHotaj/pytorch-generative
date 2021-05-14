"""Base classes for models."""

import abc

import torch
from torch import distributions
from torch import nn


def _default_sample_fn(logits):
    return distributions.Bernoulli(logits=logits).sample()


class GenerativeModel(abc.ABC, nn.Module):
    """Base class inherited by all generative models in pytorch-generative.

    Provides:
        * An abstract `sample()` method which is implemented by subclasses that support
          generating samples.
        * Variables `self._c, self._h, self._w` which store the shape of the (first)
          image Tensor the model was trained with. Note that `forward()` must have been
          called at least once and the input must be an image for these variables to be
          available.
        * A `device` property which returns the device of the model's parameters.
    """

    def __call__(self, *args, **kwargs):
        if getattr(self, "_c", None) is None and len(args[0].shape) == 4:
            _, self._c, self._h, self._w = args[0].shape
        return super().__call__(*args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    @abc.abstractmethod
    def sample(self, n_samples):
        ...


class AutoregressiveModel(GenerativeModel):
    """The base class for Autoregressive generative models. """

    def __init__(self, sample_fn=None):
        """Initializes a new AutoregressiveModel instance.

        Args:
            sample_fn: A fn(logits)->sample which takes sufficient statistics of a
                distribution as input and returns a sample from that distribution.
                Defaults to the Bernoulli distribution.
        """
        super().__init__()
        self._sample_fn = sample_fn or _default_sample_fn

    def _get_conditioned_on(self, n_samples, conditioned_on):
        assert (
            n_samples is not None or conditioned_on is not None
        ), 'Must provided one, and only one, of "n_samples" or "conditioned_on"'
        if conditioned_on is None:
            shape = (n_samples, self._c, self._h, self._w)
            conditioned_on = (torch.ones(shape) * -1).to(self.device)
        else:
            conditioned_on = conditioned_on.clone()
        return conditioned_on

    # TODO(eugenhotaj): This function does not handle subpixel sampling correctly.
    def sample(self, n_samples=None, conditioned_on=None):
        """Generates new samples from the model.

        Args:
            n_samples: The number of samples to generate. Should only be provided when
                `conditioned_on is None`.
            conditioned_on: A batch of partial samples to condition the generation on.
                Only dimensions with values < 0 are sampled while dimensions with
                values >= 0 are left unchanged. If 'None', an unconditional sample is
                generated.
        """
        with torch.no_grad():
            conditioned_on = self._get_conditioned_on(n_samples, conditioned_on)
            n, c, h, w = conditioned_on.shape
            for row in range(h):
                for col in range(w):
                    out = self.forward(conditioned_on)[:, :, row, col]
                    out = self._sample_fn(out).view(n, c)
                    conditioned_on[:, :, row, col] = torch.where(
                        conditioned_on[:, :, row, col] < 0,
                        out,
                        conditioned_on[:, :, row, col],
                    )
            return conditioned_on

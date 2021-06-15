"""Implementations of various mixture models."""

import abc

import numpy as np
import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
from pytorch_generative.models import base


class MixtureModel(base.GenerativeModel):
    """Base class inherited by all mixture models in pytorch-generative.

    Provides:
        * A generic `forward()` method which returns the log likelihood of the input
          under the distribution.` The log likelihood of the component distributions
          must be defined by the subclasses via `_component_log_prob()`.
        * A generic `sample()` method which returns samples from the distribution.
          Samples from the component distribution must be defined by the subclasses via
          `_component_sample()`.
    """

    def __init__(self, n_components, n_features):
        """Initializes a new MixtureModel instance.

        Args:
            n_components: The number of component distributions.
            n_features: The number of features (i.e. dimensions) in each component.
        """
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.mixture_logits = nn.Parameter(torch.ones((n_components,)))

    @abc.abstractmethod
    def _component_log_prob(self):
        """Returns the log likelihood of the component distributions."""

    def __call__(self, *args, **kwargs):
        x = args[0]
        self._original_shape = x.shape
        x = x.view(self._original_shape[0], 1, self.n_features)
        args = (x, *args[1:])
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        mixture_log_prob = torch.log_softmax(self.mixture_logits, dim=-1)
        log_prob = mixture_log_prob + self._component_log_prob(x)
        return torch.logsumexp(log_prob, dim=-1)

    @abc.abstractmethod
    def _component_sample(self, idxs):
        """Returns samples from the component distributions conditioned on idxs."""

    def sample(self, n_samples):
        with torch.no_grad():
            shape = (n_samples,)
            idxs = distributions.Categorical(logits=self.mixture_logits).sample(shape)
            sample = self._component_sample(idxs)
            return sample.view(n_samples, *self._original_shape[1:])


class GaussianMixtureModel(MixtureModel):
    """A categorical mixture of Gaussian distributions with diagonal covariance."""

    def __init__(self, n_components, n_features):
        super().__init__(n_components, n_features)
        self.mean = nn.Parameter(torch.randn(n_components, n_features) * 0.01)
        # NOTE: We initialize var = 1 <=> log(sqrt(var)) = 0.
        self.log_std = nn.Parameter(torch.zeros(n_components, n_features))

    def _component_log_prob(self, x):
        z = -self.log_std - 0.5 * torch.log(torch.tensor(2 * np.pi))
        log_prob = (
            z - 0.5 * ((x.unsqueeze(dim=1) - self.mean) / self.log_std.exp()) ** 2
        )
        return log_prob.sum(-1)

    def _component_sample(self, idxs):
        mean, std = self.mean[idxs], self.log_std[idxs].exp()
        return distributions.Normal(mean, std).sample()


class BernoulliMixtureModel(MixtureModel):
    """A categorical mixture of Bernoulli distributions."""

    def __init__(self, n_components, n_features):
        super().__init__(n_components, n_features)
        self.logits = nn.Parameter(torch.rand(n_components, n_features))

    def _component_log_prob(self, x):
        logits, x = torch.broadcast_tensors(self.logits, x)
        # binary_cross_entorpy_with_logits is equivalent to log Bern(x | p).
        return -F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1)

    def _component_sample(self, idxs):
        logits = self.logits[idxs]
        return distributions.Bernoulli(logits=logits).sample()

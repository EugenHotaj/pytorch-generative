import abc

import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F

# TODO(eugenhotaj): Add docs, tests, etc.


class MixtureModel(abc.ABC, nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.mixture_logits = nn.Parameter(torch.ones((n_components,)))

    @abc.abstractmethod
    def _component_log_prob(self):
        """Must be overridden by the subclass."""

    def __call__(self, x):
        self._original_shape = x.shape
        x = x.view(self._original_shape[0], 1, self.n_features)
        return super().__call__(x)

    def forward(self, x):
        mixture_log_prob = torch.log_softmax(self.mixture_logits, dim=-1)
        log_prob = mixture_log_prob + self._component_log_prob(x)
        return torch.logsumexp(log_prob, dim=-1)

    @abc.abstractmethod
    def _component_sample(self, idxs):
        """Must be overrideen by the subclass."""

    def sample(self, n_samples):
        with torch.no_grad():
            shape = (n_samples,)
            idxs = distributions.Categorical(logits=self.mixture_logits).sample(shape)
            sample = self._component_sample(idxs)
            return sample.view(n_samples, *self._original_shape[1:])


class GaussianMixtureModel(MixtureModel):
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

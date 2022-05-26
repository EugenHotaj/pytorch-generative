"""Implementation of Kernel Density Estimation (KDE) [1].

Kernel density estimation is a nonparameteric density estimation method. It works by
placing kernels K on each point in a "training" dataset D. Then, for a test point x, 
p(x) is estimated as p(x) = 1 / |D| \sum_{x_i \in D} K(u(x, x_i)), where u is some 
function of x, x_i. In order for p(x) to be a valid probability distribution, the kernel
K must also be a valid probability distribution.

References (used throughout the file):
    [1]: https://en.wikipedia.org/wiki/Kernel_density_estimation
"""

import abc

import numpy as np
import torch
from torch import nn

from pytorch_generative.models import base


class Kernel(abc.ABC, nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bandwidth=1.0):
        """Initializes a new Kernel.

        Args:
            bandwidth: The kernel's (band)width.
        """
        super().__init__()
        self.bandwidth = bandwidth

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    @abc.abstractmethod
    def forward(self, test_Xs, train_Xs):
        """Computes log p(x) for each x in test_Xs given train_Xs."""

    @abc.abstractmethod
    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class ParzenWindowKernel(Kernel):
    """Implementation of the Parzen window kernel."""

    def forward(self, test_Xs, train_Xs):
        abs_diffs = torch.abs(self._diffs(test_Xs, train_Xs))
        dims = tuple(range(len(abs_diffs.shape))[2:])
        dim = np.prod(abs_diffs.shape[2:])
        inside = torch.sum(abs_diffs / self.bandwidth <= 0.5, dim=dims) == dim
        coef = 1 / self.bandwidth ** dim
        return torch.log((coef * inside).mean(dim=1))

    def sample(self, train_Xs):
        device = train_Xs.device
        noise = (torch.rand(train_Xs.shape, device=device) - 0.5) * self.bandwidth
        return train_Xs + noise


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        n, d = train_Xs.shape
        n, h = torch.tensor(n, dtype=torch.float32), torch.tensor(self.bandwidth)
        pi = torch.tensor(np.pi)

        Z = 0.5 * d * torch.log(2 * pi) + d * torch.log(h) + torch.log(n)
        diffs = self._diffs(test_Xs, train_Xs) / h
        log_exp = -0.5 * torch.norm(diffs, p=2, dim=-1) ** 2

        return torch.logsumexp(log_exp - Z, dim=-1)

    def sample(self, train_Xs):
        device = train_Xs.device
        noise = torch.randn(train_Xs.shape) * self.bandwidth
        return train_Xs + noise


class KernelDensityEstimator(base.GenerativeModel):
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel=None):
        """Initializes a new KernelDensityEstimator.

        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.kernel = kernel or GaussianKernel()
        self.train_Xs = train_Xs
        assert len(self.train_Xs.shape) == 2, "Input cannot have more than two axes."

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an
    # iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(len(self.train_Xs)), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])

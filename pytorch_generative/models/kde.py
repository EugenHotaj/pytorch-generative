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

# TODO(ehotaj): Add tests + support NCHW tensors.


class _Kernel(abc.ABC, nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, h=0.05):
        super().__init__()
        self.h = h

    def _all_diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, test_Xs.shape[1])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], train_Xs.shape[1])
        return test_Xs - train_Xs

    def forward(self, test_Xs, train_Xs):
        """Computes p(x) for each x in test_Xs given train_Xs."""


class ParzenWindowKernel(_Kernel):
    """Implementation of the Parzen window kernel."""

    def forward(self, test_Xs, train_Xs):
        dim = test_Xs.shape[1]
        all_difs = self._all_diffs(test_Xs, train_Xs)
        n_inside = torch.sum(torch.abs(all_difs) / self.h <= 0.5, dim=2) == dim
        coef = 1 / self.h ** dim
        return (coef * n_inside).mean(dim=1)

    def sample(self, train_Xs):
        return train_Xs + (torch.rand(train_Xs) - 0.5) * self.h


class GaussianKernel(_Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        all_diffs = self._all_diffs(test_Xs, train_Xs)
        exp = torch.exp(-torch.norm(all_diffs, p=2, dim=2) ** 2 / (2 * self.h))
        coef = 1 / torch.sqrt(torch.tensor(2 * np.pi * self.h))
        return (coef * exp).mean(dim=1)

    def sample(self, train_Xs):
        return train_Xs + torch.randn(tran_Xs.shape) * self.h


# TODO(eugenhotaj): Subclass base.GenerativeModel once we support NCHW tensors.
class KernelDensityEstimator(nn.Module):
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

    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(len(self.train_Xs)), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])

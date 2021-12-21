"""Implementation of the Gaussian process model.

Gaussian processes are non-parametric models. They model the target function as a
stochastic process where each variable in the process is a (multivariate) Gaussian 
defined by mean and (co)variance functions: p(y | x) = N(y | u(x), K(x, x)). The 
variables are related via the covariance function, with the whole process being defined 
as p(f) = N(f | u(x), K(x, x').
"""

import torch
from torch import nn
from pytorch_generative.models import base


# TODO(ehotaj): Generalize to multiple dimensions.
class GaussianProcess(base.GenerativeModel):
    """The Gaussian process model."""

    # TODO(eugenhotaj): Learn the observation noise from the training data.
    def __init__(self, mean, kernel, noise_var=None):
        """Initializes a new GaussianProcess.

        Args:
            mean: Prior mean function mu(x).
            kernel: Prior covariance function K(x, x').
            noise_var: The variance of the observation noise. If not provided,
                observations are assumed to be noiseless.
        """
        super().__init__()
        self.mean = mean
        self.kernel = kernel
        self.register_buffer("noise_var", torch.tensor(noise_var or 0.0))
        self.train_x = None
        self.train_y = None

    def fit(self, x, y):
        """Fits the Gaussian process on the given training data."""
        if self.train_x is None:
            self.train_x, self.train_y = x, y
        else:
            self.train_x = torch.cat([self.train_x, x])
            self.train_y = torch.cat([self.train_y, y])

    # TODO(eugenhotaj): Figure out why PyTorch claims the covariance matrix is not
    # positive semidefinite.
    def sample(self, x, n_samples):
        """Samples n_samples from the Gaussian process at the given location.

        The samples are drawn from the posterior if `fit()` has already been called,
        otherwise the samples are drawn from the prior.

        Args:
            x: The location to sample at.
            n_samples: The number of samples to return.
        Returns:
            The samples.
        """
        with torch.no_grad():
            mu, sig = self.predict(x)
            mu, sig = mu.numpy(), sig.numpy()
        sample = np.random.multivariate_normal(mu.squeeze(), sig, size=(n_samples,))
        return torch.tensor(sample)

    def predict(self, x):
        """Computes the predicted means and variances at the given location.

        The posterior means and variances are returned if `fit()` has already been
        called otherwise the prior means and variances are returned.

        Args:
            x: The location to predict at.
        Returns
            The predicted means and variances.
        """
        if self.train_x is None:
            return self.mean(x), self.kernel(x, x)

        # Compute means and covariances.
        train_mu, x_mu = self.mean(self.train_x), self.mean(x)
        train_sig = self.kernel(
            self.train_x, self.train_x
        ) + self.noise_var * torch.eye(self.train_x.shape[0])
        x_sig, cross_sig = self.kernel(x, x), self.kernel(self.train_x, x)

        # Compute posterior mean and covariance.
        solved = torch.linalg.solve(train_sig, cross_sig).T
        mu = x_mu + solved @ (self.train_y - train_mu)
        sig = x_sig - (solved @ cross_sig)

        return mu, sig

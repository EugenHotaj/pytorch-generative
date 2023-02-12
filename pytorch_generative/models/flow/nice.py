"""Implementation of the Non-linear Independent Components Estimation (NICE) [1].

References (used throughout the code):
  [1]: https://arxiv.org/abs/1410.8516
"""

import operator

import torch
from torch import nn

from pytorch_generative.models import base


class AdditiveCouplingBlock(nn.Module):
    """Coupling block with an additive coupling law.

    Given inputs x1, x2 = split(x, 2) and coupling network m(.), the reverse mapping is
    defined as y1 = x1, y2 = x2 + m(x1) and the forward mapping is defined as
    x1 = y1, x2 = y2 - m(y1).
    """

    def __init__(self, n_features, n_hidden_layers, n_hidden_features, reverse):
        """Initializes a new AdditiveCouplingBlock instance.

        Args:
            n_features: Number of input and output features.
            n_hidden_layers: Number of hidden layers in the coupling network.
            n_hidden_features: Number of features in each hidden layer.
            reverse: Whether to reverse which half of the input is transformed by the
                coupling network.
        """
        super().__init__()
        self.reverse = reverse
        half_features = n_features // 2
        net = [
            nn.Linear(in_features=half_features, out_features=n_hidden_features),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            net.append(
                nn.Linear(in_features=n_hidden_features, out_features=n_hidden_features)
            )
            net.append(nn.ReLU())
        net.append(nn.Linear(in_features=n_hidden_features, out_features=half_features))
        self.net = nn.Sequential(*net)

    def _couple(self, x, op):
        c = x.shape[1]
        h1, h2 = x[:, : c // 2], x[:, c // 2 :]
        if self.reverse:
            h1 = op(h1, self.net(h2))
        else:
            h2 = op(h2, self.net(h1))
        return torch.cat((h1, h2), dim=1)

    def forward(self, x):
        """Inverse mapping from the inputs to the prior (X -> Z)."""
        return self._couple(x, operator.add)

    def inverse(self, y):
        """Forward mapping from the prior to the input (Z -> X)."""
        return self._couple(y, operator.sub)


class ScalingLayer(nn.Module):
    """Invertible scaling layer representing a diagonal scaling matrix S."""

    def __init__(self, n_features):
        """Initializes a new ScalingLayer instance.

        Args:
            n_features: Number of input and output features.
        """
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros((1, n_features)))

    def log_det_J(self):
        """Returns the log determinant of the Jacobian of the scaling matrix S.

        Makes use of the identity log det(S) = Tr(log S).
        """
        return torch.sum(self.log_scale)

    def _couple(self, x, sign):
        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        h = x * torch.exp(sign * self.log_scale)
        return h.view(original_shape)

    def forward(self, x):
        """Inverse mapping from the inputs to the prior (X -> Z)."""
        return self._couple(x, 1)

    def inverse(self, y):
        """Forward mapping from the prior to the input (Z -> X)."""
        return self._couple(y, -1)


class NICE(base.GenerativeModel):
    """Non-linear Independent Component Estimation (NICE) model."""

    def __init__(
        self, n_features, n_coupling_blocks=4, n_hidden_layers=5, n_hidden_features=1000
    ):
        """Initializes a new NICE instance.

        Args:
            n_features: Number of input and output features.
            n_coupling_blocks: Number of coupling blocks. Should be at least 3 to allow
                all dimensions to influence one another.
            n_hidden_layers: Number of hidden layers per coupling block.
            n_hidden_features: Number of features in each hidden layer.
        """
        super().__init__()
        net = []
        reverse = False
        for _ in range(n_coupling_blocks):
            net.append(
                AdditiveCouplingBlock(
                    n_features=n_features,
                    n_hidden_layers=n_hidden_layers,
                    n_hidden_features=n_hidden_features,
                    reverse=reverse,
                ),
            )
            reverse = not reverse
        self.net = nn.Sequential(*net)
        self.scaling = ScalingLayer(n_features)

    def forward(self, x):
        """Inverse mapping from the inputs to the prior (X -> Z)."""
        return self._forward(x), self.scaling.log_det_J()

    @base.auto_reshape
    def _forward(self, x):
        y = self.net(x)
        y = self.scaling(y)
        return y

    def sample(self, n_samples, temp=1.0):
        """See the base class.

        Args:
            n_samples: Number of samples to generate.
            temp: What temperature to use when sampling. Lower temperature produces
                higher quality samples with lower diversity.
        """
        x = torch.randn((n_samples, self._c, self._h, self._w)) * temp
        return self._inverse(x)

    @base.auto_reshape
    def _inverse(self, x):
        x = self.scaling.inverse(x)
        for block in reversed(self.net):
            x = block.inverse(x)
        return x

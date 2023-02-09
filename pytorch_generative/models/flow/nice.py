"""Implementation of the Non-linear Independent Components Estimation (NICE) [1].

TODO(eugenhotaj): List out modifications compared to [1].

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


class NICE(base.GenerativeModel):
    """Non-linear Independent Component Estimation (NICE) model."""

    def __init__(
        self, n_features, n_coupling_blocks=4, n_hidden_layers=5, n_hidden_features=5000
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

    @base.auto_reshape
    def forward(self, x):
        """Inverse mapping from the inputs to the prior (X -> Z)."""
        return self.net(x)

    @torch.no_grad()
    @base.auto_reshape
    def sample(self, n_samples, temp=1.0):
        """See the base class.

        Args:
            n_samples: Number of samples to generate.
            temp: What temperature to use when sampling. Lower temperature produces
                higher quality samples with lower diversity.
        """
        h = torch.randn(n_samples, self._c) * temp
        for block in reversed(self.net):
            h = block.inverse(h)
        return h

"""Implementation of the Non-linear Independent Components Estimation (NICE) [1].

References (used throughout the code):
  [1]: https://arxiv.org/abs/1410.8516
"""

import operator

import torch
from torch import nn

from pytorch_generative.models import base


class AdditiveCouplingBlock(base.GenerativeModel):
    def __init__(self, n_features, n_hidden, n_layers, reverse):
        super().__init__()
        self.reverse = reverse
        half_features = n_features // 2
        net = [
            nn.Linear(in_features=half_features, out_features=n_hidden),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 2):
            net.append(nn.Linear(in_features=n_hidden, out_features=n_hidden))
            net.append(nn.ReLU())
        net.append(nn.Linear(in_features=n_hidden, out_features=half_features))
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
        return self._couple(x, operator.add)

    def inverse(self, y):
        return self._couple(y, operator.sub)


class NICE(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers, n_coupling=4):
        super().__init__()
        net = []
        reverse = False
        for _ in range(n_coupling):
            net.append(
                AdditiveCouplingBlock(
                    n_features=n_features,
                    n_hidden=n_hidden,
                    n_layers=n_layers,
                    reverse=reverse,
                ),
            )
            reverse = not reverse
        self.net = nn.Sequential(*net[:-1])

    def forward(self, x):
        return self.net(x)

    def sample(self, n_samples, temp=1.0):
        h = torch.randn(n_samples, self._c) / temp
        for block in reversed(self.net):
            h = block.inverse(h)
        return h

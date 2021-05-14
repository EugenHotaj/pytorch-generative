"""Implementation of the Fully Visible Belief Network."""

import torch
from torch import nn

from pytorch_generative.models import base


# TODO(eugenhotaj): This can be sped up with masking (which is equivalent to MADE).
class FullyVisibleBeliefNetwork(base.AutoregressiveModel):
    """The Fully Visible Belief Network."""

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

        # NOTE: We use in_features=1 and always pass an input of 0 for the first Linear
        # model because PyTorch does not allow in_features=0.
        self._net = ModuleList(
            nn.Linear(in_feautres=max(1, i), out_features=1)
            for i in range(self.n_features)
        )

    def forward(self, x):
        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        output = [self._net[0](torch.zeros(original_shape[0], 1))]
        for i in range(1, self.n_features):
            output.append(self._net[i](x[:, :i]))
        output = torch.stack(output, axis=1)
        return output.view(original_shape)

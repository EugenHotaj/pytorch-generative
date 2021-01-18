"""Various utilities for building generative neural networks.

References (used throughout the code):
    [1]: https://arxiv.org/abs/1711.00937
    [2]: https://arxiv.org/abs/2003.04887
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


# TODO(eugenhotaj): It's strange that this module returns a loss.
# TODO(eugenhotaj): Move this module out of utils.
class VectorQuantizer(nn.Module):
    """A vector quantizer as introduced in [1].

    Inputs are quantized to the closest embedding in Euclidian distance. The
    embeddings can be updated using either exponential moving averages or gradient
    descent.
    """

    def __init__(self, n_embeddings, embedding_dim, use_ema=True, ema_decay=0.99):
        """Initializes a new VectorQuantizer instance.

        Args:
            n_embeddings: The number of embedding vectors. Controls the capacity in the
                information bottleneck.
            embedding_dim: Dimension of each embedding vector. Does not directly affect
                the capacity in the information bottleneck.
            use_ema: Whether to use exponential moving averages (EMA) to update the
                embedding weights instead of gradient descent. Generally, EMA updates
                lead to much faster convergence.
            ema_decay: Decay rate for exponential moving average parameters.
        """
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self._use_ema = use_ema
        self._decay = ema_decay

        embedding = torch.zeros(n_embeddings, embedding_dim)
        # TODO(eugenhotaj): Small optimization: create pre-initialized embedding.
        init.kaiming_uniform_(embedding, nonlinearity="linear")
        if self._use_ema:
            self.register_buffer("_embedding", embedding)
            self.register_buffer("_cluster_size", torch.zeros(n_embeddings))
            self.register_buffer("_embedding_avg", embedding.clone())
        else:
            self._embedding = nn.Parameter(embedding)

    def forward(self, x):
        n, c, h, w = x.shape
        assert c == self.embedding_dim, "Input channels must equal embedding_dim."

        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        # Efficient L2 distance computation which does not require materializing the
        # huge NWH * n_embeddings * embedding_dim matrix. The computation follows
        # straightforwardly from Euclidian distance definition. For more info, see
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html.
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding ** 2, dim=1)
            - 2 * flat_x @ self._embedding.t()
        )

        # Quantize to closest embedding vector.
        idxs = torch.argmin(distances, dim=1, keepdim=True)
        one_hot = torch.zeros(
            idxs.shape[0], self.n_embeddings, device=self._embedding.device
        )
        one_hot.scatter_(1, idxs, 1)
        quantized = one_hot @ self._embedding
        quantized = quantized.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()

        # NOTE: Most implementations weight the commitment loss by some constant
        # given by the user. However, we find a weight of 1 is quite robust.
        loss = F.mse_loss(x, quantized.detach())  # Commitment loss.
        if self._use_ema and self.training:
            batch_cluster_size = one_hot.sum(axis=0)
            batch_embedding_avg = (flat_x.t() @ one_hot).t()
            self._cluster_size.data.mul_(self._decay).add_(
                batch_cluster_size, alpha=1 - self._decay
            )
            self._embedding_avg.data.mul_(self._decay).add_(
                batch_embedding_avg, alpha=1 - self._decay
            )
            new_emb = self._embedding_avg / (self._cluster_size + 1e-5).unsqueeze(1)
            self._embedding.data.copy_(new_emb)
        elif not self._use_ema:
            # Add the embedding loss when not using EMA.
            loss += F.mse_loss(quantized, x.detach())

        quantized = x + (quantized - x).detach()  # Straight through estimator.
        return quantized, loss


class ReZeroWrapper(nn.Module):
    """Wraps a given module into a ReZero [2] function.

    ReZero computes `x + alpha * module(x)` for some input `x`. `alpha` is a trainable
    scalar parameter which is initialized to `0`. Note that `module(x)` must have the
    same output shape as the input `x`.
    """

    def __init__(self, module):
        """Initializes a new ReZeroWrapper.

        Args:
            module: The module to wrap.
        """
        self._module = module
        self._alpha = nn.Parameter(torch.tensor([0.0]))

    def forward(x):
        return x + self._alpha * self._module(x)

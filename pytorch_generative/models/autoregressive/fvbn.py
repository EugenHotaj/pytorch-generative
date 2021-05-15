"""Implementation of the Fully Visible Belief Network (FVBN) [1].

The FVBN is an autoregressive model composed of a collection of linear models. Each
linear model tries to model p(x_i|x_{j<i}). The FVBN can be viewed as a special case of
MADE [2] with no hidden layers and one set of masks applied in raster-scan order.

References (used throughout code):
    [1]: https://www.semanticscholar.org/paper/Connectionist-Learning-of-Belief-Networks-Neal/a120c05ad7cd4ce2eb8fb9697e16c7c4877208a5
    [2]: https://arxiv.org/pdf/1502.03509.pdf
"""

import torch
from torch import nn

from pytorch_generative.models import base


# TODO(eugenhotaj): This can be sped up with masking (which is equivalent to MADE).
class FullyVisibleBeliefNetwork(base.AutoregressiveModel):
    """The Fully Visible Belief Network."""

    def __init__(self, n_dims):
        """Initializes a new FullyVisibleBeliefNetwork.

        Args:
            n_dims: Number of input (and output) dimensions.
        """
        super().__init__()
        self.n_dims = n_dims

        # NOTE: We use in_features=1 and always pass an input of 0 for the first Linear
        # model because PyTorch does not allow in_features=0.
        self._net = nn.ModuleList(
            nn.Linear(in_features=max(1, i), out_features=1) for i in range(self.n_dims)
        )

    def forward(self, x):
        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        output = [self._net[0](torch.zeros(original_shape[0], 1, device=self.device))]
        for i in range(1, self.n_dims):
            output.append(self._net[i](x[:, :i]))
        output = torch.stack(output, axis=1)
        return output.view(original_shape)


def reproduce(
    n_epochs=50,
    batch_size=512,
    log_dir="/tmp/run",
    n_gpus=1,
    device_id=0,
    debug_loader=None,
):
    """Training script with defaults to reproduce results.
    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.
    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        n_gpus: Number of GPUs to use for training the model. If 0, uses CPU.
        device_id: The device_id of the current GPU when training on multiple GPUs.
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """
    from torch import optim
    from torch.nn import functional as F

    from pytorch_generative import datasets
    from pytorch_generative import models
    from pytorch_generative import trainer

    train_loader, test_loader = debug_loader, debug_loader
    if train_loader is None:
        train_loader, test_loader = datasets.get_mnist_loaders(
            batch_size, dynamically_binarize=True
        )

    model = models.FullyVisibleBeliefNetwork(n_dims=784)
    optimizer = optim.Adam(model.parameters())

    def loss_fn(x, _, preds):
        loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none").sum()
        return loss / x.shape[0]

    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        log_dir=log_dir,
        n_gpus=n_gpus,
        device_id=device_id,
    )
    model_trainer.interleaved_train_and_eval(n_epochs)

"""Base classes for models."""

import abc

import torch
from torch import distributions, nn


def _default_sample_fn(logits):
    return distributions.Bernoulli(logits=logits).sample()


def auto_reshape(fn):
    """Decorator which flattens image inputs and reshapes them before returning.

    This is used to enable non-convolutional models to transparently work on images.
    """

    def wrapped_fn(self, x, *args, **kwargs):
        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        y = fn(self, x, *args, **kwargs)
        return y.view(original_shape)

    return wrapped_fn


class GenerativeModel(abc.ABC, nn.Module):
    """Base class inherited by all generative models in pytorch-generative.

    Provides:
        * An abstract `sample()` method which is implemented by subclasses that support
          generating samples.
        * Variables `self._c, self._h, self._w` which store the shape of the (first)
          image Tensor the model was trained with. Note that `forward()` must have been
          called at least once and the input must be an image for these variables to be
          available.
        * A `device` property which returns the device of the model's parameters.
    """

    def __call__(self, x, *args, **kwargs):
        """Saves input tensor attributes so they can be accessed during sampling."""
        if getattr(self, "_c", None) is None and x.dim() == 4:
            _, c, h, w = x.shape
            self._create_shape_buffers(c, h, w)
        return super().__call__(x, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Registers dynamic buffers before loading the model state."""
        if "_c" in state_dict and not getattr(self, "_c", None):
            c, h, w = state_dict["_c"], state_dict["_h"], state_dict["_w"]
            self._create_shape_buffers(c, h, w)
        super().load_state_dict(state_dict, strict)

    def _create_shape_buffers(self, channels, height, width):
        channels = channels if torch.is_tensor(channels) else torch.tensor(channels)
        height = height if torch.is_tensor(height) else torch.tensor(height)
        width = width if torch.is_tensor(width) else torch.tensor(width)
        self.register_buffer("_c", channels)
        self.register_buffer("_h", height)
        self.register_buffer("_w", width)

    @property
    def device(self):
        return next(self.parameters()).device

    @abc.abstractmethod
    def sample(self, n_samples):
        ...


class AutoregressiveModel(GenerativeModel):
    """The base class for Autoregressive generative models."""

    def __init__(self, sample_fn=None):
        """Initializes a new AutoregressiveModel instance.

        Args:
            sample_fn: A fn(logits)->sample which takes sufficient statistics of a
                distribution as input and returns a sample from that distribution.
                Defaults to the Bernoulli distribution.
        """
        super().__init__()
        self._sample_fn = sample_fn or _default_sample_fn

    def _get_conditioned_on(self, n_samples, conditioned_on):
        assert (
            n_samples is not None or conditioned_on is not None
        ), 'Must provided one, and only one, of "n_samples" or "conditioned_on"'
        if conditioned_on is None:
            shape = (n_samples, self._c, self._h, self._w)
            conditioned_on = (torch.ones(shape) * -1).to(self.device)
        else:
            conditioned_on = conditioned_on.clone()
        return conditioned_on

    @torch.no_grad()
    def sample(self, n_samples=None, conditioned_on=None):
        """Generates new samples from the model.

        Args:
            n_samples: The number of samples to generate. Should only be provided when
                `conditioned_on is None`.
            conditioned_on: A batch of partial samples to condition the generation on.
                Only dimensions with values < 0 are sampled while dimensions with
                values >= 0 are left unchanged. If 'None', an unconditional sample is
                generated.
        """
        conditioned_on = self._get_conditioned_on(n_samples, conditioned_on)
        n, c, h, w = conditioned_on.shape
        for row in range(h):
            for col in range(w):
                out = self.forward(conditioned_on)[:, :, row, col]
                out = self._sample_fn(out).view(n, c)
                conditioned_on[:, :, row, col] = torch.where(
                    conditioned_on[:, :, row, col] < 0,
                    out,
                    conditioned_on[:, :, row, col],
                )
        return conditioned_on


class VariationalAutoEncoder(GenerativeModel):
    def __init__(self, sample_fn=None):
        super().__init__()
        self._sample_fn = sample_fn or _default_sample_fn

    @abc.abstractmethod
    def _sample(self, n_samples):
        ...

    @torch.no_grad()
    def sample(self, n_samples):
        return self._sample_fn(self._sample(n_samples))

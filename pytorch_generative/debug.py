"""Utilities for debugging models in PyTorch."""

import torch
from torch.utils.data import dataset


def compute_receptive_field(model, img_size=(1, 3,  3)):
  """Computes the receptive field for a model.

  The receptive field is computed using the magnitude of the gradient of the
  model's output with respect to the input.

  Args:
    model: Model for hich to compute the receptive field. Assumes NCHW input.
    img_size: The (channels, height, width) of the input to the model.
  """
  c, h, w = img_size
  img = torch.randn((1, c, h, w), requires_grad=True)
  model(img)[0, 0, h//2, w//2].mean().backward()
  grad = img.grad.abs()[0, 0, :, :]
  return torch.where(
      grad > 0, torch.ones_like(grad), torch.zeros_like(grad))


class OneExampleLoaderWrapper:
  """A torch.utils.DataLoader wrapper which always returns the same example."""

  def __init__(self, loader):
    """Initializes a new OneBatchLoaderWrapper instance.

    Args:
      loader: The torch.utils.DataLoader to wrap. We assume the loader returns
        tuples of batches where each item in the tuple has batch_size as
        the first dimension. We do not impose a restriction on the size of the
        tuple. E.g., (X), (X, Y), (X, Y, Z), ... are all valid tuples as long as
        X.shape[0] == Y.shape[0] == Z.shape[0] == batch_size.
    """
    self._exhausted = False
    batch = next(iter(loader))
    self.dataset = dataset.TensorDataset(*[item[:1] for item in batch])

  def __iter__(self):
    self._exhausted = False
    return self

  def __next__(self):
    if not self._exhausted:
      self._exhausted = True
      return self.dataset[:]
    raise StopIteration()


class OneBatchLoaderWrapper:
  """A torch.utils.DataLoader wrapper which always returns the same batch."""

  def __init__(self, loader):
    """Initializes a new OneBatchLoaderWrapper instance.
    
    Args:
      loader: The torch.utils.DataLoader to wrap.
    """
    self._exhausted = False
    self.dataset = dataset.TensorDataset(*next(iter(loader)))

  def __iter__(self):
    self._exhausted = False
    return self

  def __next__(self):
    if not self._exhausted:
      self._exhausted = True
      return self.dataset[:]
    raise StopIteration()

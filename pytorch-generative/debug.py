"""Utilities for debugging models in PyTorch."""


class OneBatchLoaderWrapper:
  """A torch.utils.DataLoader wrapper which always returns the same batch."""

  def __init__(self, loader):
    """Initializes a new OneBatchLoaderWrapper instance.
    
    Args:
      loader: The torch.utils.DataLoader to wrap.
    """
    self._exhausted = False
    self._batch = next(iter(loader))

  def __iter__(self):
    self._exhausted = False
    return self

  def __next__(self):
    if not self._exhausted:
      self._exhausted = True
      return self._batch
    raise StopIteration() 


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
    self._example = [item[:1] for item in batch]

  def __iter__(self):
    self._exhausted = False
    return self

  def __next__(self):
    if not self._exhausted:
      self._exhausted = True
      return self._example
    raise StopIteration() 

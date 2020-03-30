"""Utilities to make life easier when working with Google Colaboratory.

Warning: This module must be imported from Colab, otherwise it will crash.
"""

import collections
import os

import PIL
from google.colab import files
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def get_device():
  """Returns the appropriate device depending on what's available."""
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def upload_files():
  """Creates a widget to upload files from your local machine to Colab.

  The files are saved in '/tmp/<file_name>'.
  """
  uploaded = files.upload()
  for name, data in uploaded.items():
    with open(f'/tmp/{name}', 'wb') as f:
      f.write(data)


def load_image(path, size=None, remove_alpha_channel=True):
  """Loads an image from the given path as a torch.Tensor.

  Args:
    path: The path to the image to load.
    size: Either None, an integer, or a pair of integers. If not None, the 
      image is resized to the given size before being returned.
    remove_alpha_channel: If True, removes the alpha channel from the image.
  Returns:
    The loaded image as a torch.Tensor.
  """ 
  transform = []
  if size is not None:
    size = size if isinstance(size, collections.Sequence) else (size, size)
    assert len(size) == 2, "'size' must either be a scalar or contain 2 items"
    transform.append(transforms.Resize(size))
  transform.append(transforms.ToTensor())
  image_loader = transforms.Compose(transform)

  image = PIL.Image.open(path)
  image = image_loader(image)
  if remove_alpha_channel:
      image = image[:3, :, :]
  image = image.to(torch.float)

  return image
  

def imshow(tensor, title=None, figsize=None):
  """Renders the given tensor as an image using Matplotlib.

  Args:
    tensor: The tensor to render as an image.
    title: The title for the rendered image. Passed to Matplotlib.
    figsize: The size (in inches) for the image. Passed to Matplotlib.
  """
  image_unloader = transforms.ToPILImage()
  tensor = tensor.cpu().clone().squeeze(0)
  image = image_unloader(tensor)

  plt.figure(figsize=figsize)
  plt.title(title)
  plt.axis('off')
  plt.imshow(image)

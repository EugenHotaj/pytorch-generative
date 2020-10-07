"""Utilities to make life easier when working with Google Colaboratory.

Warning: This module must be imported from Colab, otherwise it will crash.
"""

import collections
import time
import os

import PIL
from google.colab import files
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Always use html5 for animations so they can be rendered inline on Colab.
matplotlib.rcParams['animation.html'] = 'html5'

_IMAGE_UNLOADER = transforms.Compose([
  transforms.Lambda(lambda x: x.cpu().clone().squeeze(0)),
  transforms.ToPILImage()
])


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
  

def imshow(batch_or_tensor, title=None, figsize=None, **kwargs):
  """Renders tensors as an image using Matplotlib.

  Args:
    batch_or_tensor: A batch or single tensor to render as images. If the batch
      size > 1, the tensors are flattened into a horizontal strip before being
      rendered.
    title: The title for the rendered image. Passed to Matplotlib.
    figsize: The size (in inches) for the image. Passed to Matplotlib.
    **kwargs: Extra keyword arguments passed as pyplot.imshow(image, **kwargs).
  """
  batch = batch_or_tensor
  for _ in range(4 - batch.ndim):
    batch = batch.unsqueeze(0)
  n, c, h, w = batch.shape
  tensor = batch.permute(1, 2, 0, 3).reshape(c, h, -1)
  image = _IMAGE_UNLOADER(tensor)

  plt.figure(figsize=figsize)
  plt.title(title)
  plt.axis('off')
  plt.imshow(image, **kwargs)


def animate(frames, figsize=None, fps=24):
  """Renders the given frames together into an animation.
  
  Args:
    frames: Either a list, iterator, or generator of images in torch.Tensor 
      format.
    figsize: The display size for the animation; passed to Matplotlib.
    fps: The number of frames to render per second (i.e. frames per second).
  Returns:
    The Matplotlib animation object.
  """
  fig = plt.figure(figsize=figsize)
  fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
  plt.axis('off')

  # We pass a fake 2x2 image to 'imshow' since it does not allow None or empty
  # lists to be passed in. The fake image data is then updated by animate_fn.
  image = plt.imshow([[0, 0], [0, 0]])
  def animate_fn(frame):
    frame = _IMAGE_UNLOADER(frame)
    image.set_data(frame)
    return image,

  anim = animation.FuncAnimation(
      fig, 
      animate_fn, 
      frames=frames, 
      interval=1000 / fps,
      blit=True,
      # Caching frames causes OOMs in Colab when there are a lot of frames or 
      # the size of individual frames is large.
      cache_frame_data=False)
  plt.close(anim._fig)
  return anim

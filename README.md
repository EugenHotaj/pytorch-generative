# pytorch-generative

`pytorch-generative` is a Python library which makes generative modeling in PyTorch easier by providing:

* high quality reference implementations of SOTA generative models 
* useful abstractions of common [building blocks](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn.py) found in the literature
* utilities for training, debugging, and working with Google Colab

To get started, click on one of the links below.
* [Example - ImageGPT](#example---imagegpt)
* [Supported Algorithms](#supported-algorithms) 
  * [Autoregressive Models](#autoregressive-models)
  * [Variational Autoencoders](#variational-autoencoders)
  * [Neural Style Transfer](#neural-style-transfer)
  * [Compositional Pattern Producing Networks](compositional-pattern-producing-networks)

## Example - ImageGPT

Supported models are implemented as PyTorch Modules and are easy to use:

```python
from pytorch_generative import models

model = models.ImageGPT(in_channels=1, out_channels=1, in_size=28)
...
model(data)
```

Alternatively, lower level building blocks in [pytorch_generative.nn](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn.py) can be used to write models from scratch. For example, we implement a convolutional [ImageGPT](https://openai.com/blog/image-gpt/)-like model below:

```python

from torch import nn

from pytorch_generative import nn as pg_nn


class TransformerBlock(nn.Module):
  """An ImageGPT Transformer block."""

  def __init__(self, 
               n_channels, 
               n_attention_heads):
    """Initializes a new TransformerBlock instance.
    
    Args:
      n_channels: The number of input and output channels.
      n_attention_heads: The number of attention heads to use.
    """
    super().__init__()
    self._ln1 = pg_nn.NCHWLayerNorm(n_channels)
    self._ln2 = pg_nn.NCHWLayerNorm(n_channels)
    self._attn = pg_nn.MaskedAttention(
        in_channels=n_channels,
        embed_channels=n_channels,
        out_channels=n_channels,
        n_heads=n_attention_heads,
        is_causal=False)
    self._out = nn.Sequential(
        nn.Conv2d(
            in_channels=n_channels, 
            out_channels=4*n_channels, 
            kernel_size=1),
        nn.GELU(),
        nn.Conv2d(
            in_channels=4*n_channels, 
            out_channels=n_channels, 
            kernel_size=1))

  def forward(self, x):
    x = x + self._attn(self._ln1(x))
    return x + self._out(self._ln2(x))


class ImageGPT(nn.Module):
  """The ImageGPT Model."""
  
  def __init__(self,       
               in_channels,
               out_channels,
               in_size,
               n_transformer_blocks=8,
               n_attention_heads=4,
               n_embedding_channels=16):
    """Initializes a new ImageGPT instance.
    
    Args:
      in_channels: The number of input channels.
      out_channels: The number of output channels.
      in_size: Size of the input images. Used to create positional encodings.
      n_transformer_blocks: Number of TransformerBlocks to use.
      n_attention_heads: Number of attention heads to use.
      n_embedding_channels: Number of attention embedding channels to use.
    """
    super().__init__()
    self._pos = nn.Parameter(torch.zeros(1, in_channels, in_size, in_size))
    self._input = pg_nn.MaskedConv2d(
        is_causal=True,
        in_channels=in_channels,
        out_channels=n_embedding_channels,
        kernel_size=3,
        padding=1)
    self._transformer = nn.Sequential(
        *[TransformerBlock(n_channels=n_embedding_channels,
                         n_attention_heads=n_attention_heads)
          for _ in range(n_transformer_blocks)])
    self._ln = pg_nn.NCHWLayerNorm(n_embedding_channels)
    self._out = nn.Conv2d(in_channels=n_embedding_channels,
                          out_channels=out_channels,
                          kernel_size=1)

  def forward(self, x):
    x = self._input(x + self._pos)
    x = self._transformer(x)
    x = self._ln(x)
    return self._out(x)
```

## Supported Algorithms

 `pytorch-generative` supports the following algorithms. 

*Note:* Our reported binary MNIST results may be optimistic. Instead of using a fixed dataset, we resample a new binary MNIST dataset on every epoch. We can think of this as using data augmentation which helps our models learn better.

### Autoregressive Models

[Binarized MNIST](https://paperswithcode.com/sota/image-generation-on-binarized-mnist) (nats): 

| Algorithm | Our Results | Links |
| --- | ---| --- |
| PixelSNAIL | **78.61** | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/pixel_snail.py), [Paper](http://proceedings.mlr.press/v80/chen18h/chen18h.pdf) |
| ImageGPT | 79.17 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/image_gpt.py), [Paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)|
| Gated PixelCNN | 81.50 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/gated_pixel_cnn.py), [Paper](https://arxiv.org/pdf/1601.06759.pdf) |
| PixelCNN | 81.45 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/pixel_cnn.py), [Paper](https://arxiv.org/pdf/1601.06759.pdf) |
| MADE | 84.87 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/made.py), [Paper](https://arxiv.org/pdf/1502.03509.pdf) |
| NADE | 85.65 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/nade.py), [Paper](http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf) |

### Variational Autoencoders

| Algorithm | Our Results | Links |
| --- | ---| --- |
| VAE | TODO | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/vae.py), [Paper](https://arxiv.org/pdf/1312.6114.pdf) |
| VQ-VAE | TODO | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/vq_vae.py), [Paper](https://arxiv.org/pdf/1711.00937.pdf) |
| VQ-VAE-2 | TODO | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/vq_vae_2.py), [Paper](https://arxiv.org/pdf/1906.00446.pdf) |


### Neural Style Transfer
Blog: https://towardsdatascience.com/how-to-get-beautiful-results-with-neural-style-transfer-75d0c05d6489 <br>
Notebook: https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/style_transfer.ipynb <br>
Paper: https://arxiv.org/pdf/1508.06576.pdf

### Compositional Pattern Producing Networks
Notebook: https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/cppn.ipynb <br>
Background: https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

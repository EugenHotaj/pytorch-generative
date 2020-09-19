# pytorch-generative
**pytorch-generative** is a nascent project that aims to provide a simple, easy to use library for generative modeling in PyTorch. 

The library makes generative model implementation and experimentation easier by abstracting common building blocks such as [MaskedConv2d](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn.py#L58-L96) and [MaskedAttention](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn.py#L99-L175).
It also provides clean, high quality reference implementations of recent State of the Art papers that are easy to read, understand, and extend. 
Finally, it provides utilities for training, debugging, and working with Google Colab.

So far, the library has primarily focues on Autoregressive modeling. The future goal is to also expand into VAEs, GANS, Flows, etc. 

## Example - ImageGPT

Supported models are implemented as PyTorch Modules and are easy to use:

```python
from pytorch_generative import models

model = models.ImageGPT(in_channels=1, in_size=28)
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
    x = x + self._attn(x)
    return x + self._out(x)


class ImageGPT(nn.Module):
  """The ImageGPT Model.
  
  Note that we don't use LayerNorm because it would break the model's 
  autoregressive property.
  """
  
  def __init__(self,       
               in_channels,
               in_size,
               n_transformer_blocks=8,
               n_attention_heads=4,
               n_embedding_channels=16):
    """Initializes a new ImageGPT instance.
    
    Args:
      in_channels: The number of input channels.
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
    self._out = nn.Conv2d(in_channels=n_embedding_channels,
                          out_channels=in_channels,
                          kernel_size=1)

  def forward(self, x):
    x = self._input(x + self._pos)
    x = self._transformer(x)
    return self._out(x)
```

## Supported Algorithms

 **pytorch-generative** supports the following algorithms. 
 We train most algorithms on [Binarized Mnist](https://paperswithcode.com/sota/image-generation-on-binarized-mnist) 
 and either match or surpass the relevant papers.

### Autoregressive Generative Models

Binary MNIST (NLL): 

| Algorithm | Our Results | Best Other Results | Links |
| --- | ---| --- | --- |
| ImageGPT | TODO | N/A | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/image_gpt.py), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/image_gpt.ipynb) |
| PixelSNAIL | 78.61 | N/A | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/pixel_snail.py), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/pixel_snail.ipynb) |
| Gated PixelCNN | 81.50 | **81.30** \[1\] | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/gated_pixel_cnn.py), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/gated_pixel_cnn.ipynb) |
| PixelCNN | 81.45 | **81.30** \[1\] | [Code](), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/pixel_cnn.ipynb) |
| MADE | **84.867** | 88.04 \[4\]| [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/made.py), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/MADE.ipynb) |
| NADE | **85.65** | 88.86 \[5\] | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/nade.py), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/NADE.ipynb) |

*Note:* Our reported binary MNIST results may be optimistic. Instead of using a fixed dataset, we resample a new binary MNIST dataset on every epoch. We can think of this as using data augmentation which helps our models learn better.

#### References

1. https://arxiv.org/pdf/1601.06759.pdf 
1. https://arxiv.org/abs/1606.05328
1. http://www.scottreed.info/files/iclr2017.pdf
1. https://arxiv.org/abs/1502.03509 
1. http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf

### Neural Style Transfer
Blog: https://towardsdatascience.com/how-to-get-beautiful-results-with-neural-style-transfer-75d0c05d6489 <br>
Notebook: https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/style_transfer.ipynb <br>
Paper: https://arxiv.org/pdf/1508.06576.pdf

### Compositional Pattern Producing Networks
Notebook: https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/cppn.ipynb <br>
Background: https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

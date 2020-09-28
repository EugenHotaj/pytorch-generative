"""Implementation of (convolutional) ImageGPT.

ImageGPT is an autoregressive generative model which uses a (decoder only)
Transformer architecture for image generation.

N.B.: Our implementation operates over images instead of embedding tokens like 
[1]. This defeats the purpose slightly as the main motivation of the original 
paper is to demonstrate that the same architecture can be effective for both 
images and text. Our intention, instead, is to demonstrate the capabilities of
the pytorch-generative library.

References (used throughout the file):
  [1]: https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf
"""

import torch
from torch import distributions
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base 


class TransformerBlock(nn.Module):
  """An ImageGPT Transformer block."""

  def __init__(self, 
               n_channels, 
               n_attention_heads, 
               use_linear_attention=False):
    """Initializes a new TransformerBlock instance.

    Args:
      n_channels: The number of input and output channels.
      n_attention_heads: The number of attention heads to use.
      use_linear_attention: Whether to use LinearMaskedAttention instead of the
        regular MaskedAttention.
    """
    super().__init__()
    self._ln1 = pg_nn.NCHWLayerNorm(n_channels)
    self._ln2 = pg_nn.NCHWLayerNorm(n_channels)
    attn = (pg_nn.LinearMaskedAttention if use_linear_attention 
            else pg_nn.MaskedAttention)
    self._attn = attn(
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
        nn.Conv2d (
            in_channels=4*n_channels, 
            out_channels=n_channels, 
            kernel_size=1))

  def forward(self, x):
    x = x + self._attn(self._ln1(x))
    return x + self._out(self._ln2(x))


class ImageGPT(base.AutoregressiveModel):
  """The ImageGPT Model.
  
  Unlike [1], our implementation operates over image inputs, instead of 
  embeddings. Furthermore, we implement skip connections from each block to the 
  output. We find that this makes training a lot more stable and allows for much 
  faster convergence.
  """
  def __init__(self,       
               in_channels,
               in_size,
               out_dim=1,
               n_transformer_blocks=8,
               n_attention_heads=4,
               n_embedding_channels=16,
               use_linear_attention=False,
               sample_fn=None):
    """Initializes a new ImageGPT instance.
    
    Args:
      in_channels: The number of input channels.
      in_size: Size of the input images. Used to create positional encodings.
      out_dim: The dimension of the output. Given input of the form NCHW, the 
        output from the GatedPixelCNN model will be N(out_dim*C)HW.
      probs_fn: See the base class.
      n_transformer_blocks: Number of TransformerBlocks to use.
      n_attention_heads: Number of attention heads to use.
      n_embedding_channels: Number of attention embedding channels to use.
      use_linear_attention: Whether to use pg_nn.LinearMaskedAttention for the
        attention computation. LinearMaskedAttention is much more memory 
        efficient than MaskedAttention but is much slower to compute.
      sample_fn: See the base class.
    """
    super().__init__(sample_fn)
    self._pos = nn.Parameter(torch.zeros(1, in_channels, in_size, in_size))
    self._input = pg_nn.MaskedConv2d(
        is_causal=True, 
        in_channels=in_channels,
        out_channels=n_embedding_channels,
        kernel_size=3,
        padding=1)
    self._transformer = nn.ModuleList(
        TransformerBlock(n_channels=n_embedding_channels,
                         n_attention_heads=n_attention_heads,
                         use_linear_attention=use_linear_attention)
        for _ in range(n_transformer_blocks))
    self._ln = pg_nn.NCHWLayerNorm(n_embedding_channels)
    self._out = nn.Conv2d(in_channels=n_embedding_channels,
                          out_channels=out_dim * in_channels,
                          kernel_size=1)

  def forward(self, x):
    x = self._input(x + self._pos)
    for block in self._transformer:
      x = x + block(x)
    return self._out(self._ln(x))

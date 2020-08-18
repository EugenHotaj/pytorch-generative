"""Implementation of PixelSNAIL [1].

PixelSNAIL extends PixelCNN [2] (and its variants) by introducing a causally
masked attention layer. This layer extends the model's receptive field by 
allowing each pixel to explicitly depend on all previous pixels. PixelCNN's
receptive field, on the other hand, can only be increased by using deeper 
networks with more convolutions. The attention block also naturally resolves 
the blind spot issue which PixelCNN suffers from without needing a complex two 
stream architecture.

Unlike [1], we use skip connections from each PixelSNAILBlock to the output.
We find that this greatly stabilizes the model during training and gets rid of
exploding gradient issues. It also massively speeds up convergence.

References (used throughout the code):
  [1]: https://arxiv.org/abs/1712.09763
  [2]: https://arxiv.org/abs/1606.05328
"""

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base 


def _elu_conv_elu(conv, x):
  return F.elu(conv(F.elu(x)))


class ResidualBlock(nn.Module):
  """Residual block with a gated activation function."""
  
  def __init__(self, n_channels):
    """Initializes a new ResidualBlock.

    Args:
      n_channels: The number of input and output channels.
    """
    super().__init__()
    self._input_conv = nn.Conv2d(
        in_channels=n_channels, out_channels=n_channels, kernel_size=2, 
        padding=1)
    self._output_conv = nn.Conv2d(
        in_channels=n_channels, out_channels=2*n_channels, kernel_size=2, 
        padding=1)
    self._activation = pg_nn.GatedActivation(activation_fn=nn.Identity())

  def forward(self, x):    
    _, c, h, w = x.shape
    out = _elu_conv_elu(self._input_conv, x)[:, :, :h, :w]
    out = self._activation(self._output_conv(out)[:, :, :h, :w])
    return x + out


class PixelSNAILBlock(nn.Module):
  """Block comprised of a number of residual blocks plus one attention block.

  Implements Figure 5 of [1].
  """ 
  
  def __init__(self, 
               n_channels,
               n_residual_blocks=2,
               attention_key_channels=4,
               attention_value_channels=32,
               input_img_channels=1):
    """Initializes a new PixelSnailBlock instance.

    Args:
      n_channels: Number of input and output channels.
      n_residual_blocks: Number of residual blocks.
      attention_key_channels: Number of channels (dimension) for the attention 
        key.
      attention_value_channels: Number of channels (dimension) for the attention 
        value.
      original_input_channels: The number of channels in the original input_img.
        Used for the positional encoding channels and the extra channels for 
        the key and value convolutions in the attention block.
    """
    super().__init__()

    def conv(in_channels):
      return nn.Conv2d(in_channels, out_channels=n_channels, kernel_size=1)
       
    self._residual = nn.Sequential(
        *[ResidualBlock(n_channels) for _ in range(n_residual_blocks)])
    self._attention = pg_nn.MaskedAttention(
        query_channels=n_channels + 2 * input_img_channels, 
        key_channels=attention_key_channels,
        value_channels=attention_value_channels, 
        is_causal=True,
        kv_extra_channels=input_img_channels)
    self._residual_out = conv(n_channels)
    self._attention_out = conv(attention_value_channels)
    self._out = conv(n_channels)

  def forward(self, x, input_img):
    """Computes the forward pass.
    
    Args:
      x: The input.
      input_img: The original image only used as input to the attention blocks.
    Returns:
      The result of the forward pass.
    """
    res = self._residual(x)
    pos = pg_nn.image_positional_encoding(input_img.shape)
    attn = self._attention(torch.cat((pos, res), dim=1), input_img)
    res, attn = (_elu_conv_elu(self._residual_out, res), 
                 _elu_conv_elu(self._attention_out, attn))
    return _elu_conv_elu(self._out, res + attn)


class PixelSNAIL(base.AutoregressiveModel):
  """The PixelSNAIL model.

  Unlike [1], we implement skip connections from each block to the output.
  We find that this makes training a lot more stable and allows for much faster
  convergence.
  """

  def __init__(self, 
               in_channels=1, 
               out_dim=1,
               n_channels=64,
               n_pixel_snail_blocks=8,
               n_residual_blocks=2,
               attention_key_channels=4,
               attention_value_channels=32,
               head_channels=1):
    """Initializes a new PixelSNAIL instance.

    Args:
      in_channels: The number of channels in the input image (typically either 
        1 or 3 for black and white or color images respectively).
      out_dim: The dimension of the output. Given input of the form NCHW, the 
        output from the model will be N out_dim CHW.
      n_channels: The number of channels to use for convolutions.
      n_pixel_snail_blocks: The number of PixelSNAILBlocks. 
      n_residual_blocks: The number of ResidualBlock to use in each 
        PixelSnailBlock.
      attention_key_channels: Number of channels (dimension) for the attention 
        key.
      attention_value_channels: Number of channels (dimension) for the attention 
        value.

    """
    super().__init__()
    self._out_dim = out_dim
    
    self._input = pg_nn.MaskedConv2d(is_causal=True, 
                                     in_channels=in_channels, 
                                     out_channels=n_channels,
                                     kernel_size=3,
                                     padding=1)
    self._pixel_snail_blocks = nn.ModuleList([
        PixelSNAILBlock(n_channels=n_channels,
                        n_residual_blocks=n_residual_blocks, 
                        attention_key_channels=attention_key_channels,
                        attention_value_channels=attention_value_channels)
        for _ in range(n_pixel_snail_blocks)
    ])
    self._output = nn.Sequential(
        nn.Conv2d(
          in_channels=n_channels, out_channels=head_channels, kernel_size=1),
        nn.Conv2d(in_channels=head_channels, 
                  out_channels=self._out_dim * in_channels, 
                  kernel_size=1))

  def forward(self, x):
    n, c, h, w = x.shape

    input_img = x
    x = self._input(x)
    skip = x
    for block in self._pixel_snail_blocks:
      x = block(skip, input_img)
      skip += x
    return torch.sigmoid(self._output(skip)).view(n, self._out_dim, c, h, w)

"""TODO(eugenhotaj): Add docs."""

class MaskedConv2d(nn.Conv2d):

  def __init__(self, is_causal, *args, **kwargs):
    super().__init__(*args, **kwargs)

    i, o, h, w = self.weight.shape

    assert h % 2 == 1, 'kernel_size cannot be even'
    
    mask = torch.zeros((i, o, h, w))
    mask.data[:, :, :h//2, :] = 1
    mask.data[:, :, h//2, :w//2 + int(not is_causal)] = 1
    self.register_buffer('mask', mask)

  def forward(self, x):
    self.weight.data *= self.mask
    return super(MaskedConv2d, self).forward(x)
    
    
class MaskedResidualBlock(nn.Module):
  
  def __init__(self, n_channels):
    super().__init__()
    self._net = nn.Sequential(
        # NOTE(eugenhotaj): The PixelCNN paper users Relu->Conv2d since they do
        # not use a ReLU in the first layer. 
        nn.Conv2d(in_channels=n_channels, 
                  out_channels=n_channels//2, 
                  kernel_size=1),
        nn.ReLU(),
        MaskedConv2d(is_causal=False,
                     in_channels=n_channels//2,
                     out_channels=n_channels//2,
                     kernel_size=3,
                     padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=n_channels//2, 
                  out_channels=n_channels,
                  kernel_size=1),
        nn.ReLU())

  def forward(self, x):
    return x + self._net(x)


class PixelCNN(nn.Module):

  def __init__(self, 
               in_channels, 
               residual_channels=128, 
               head_channels=32,
               n_residual_blocks=15):
    """Initializes a new PixelCNN instance.
    
    Args:
      in_channels: The number of channels in the input image (typically either 
        1 or 3 for black and white or color images respectively).
      residual_channels: The number of channels to use in the residual layers.
      head_channels: The number of channels to use in the two 1x1 convolutional
        layers at the head of the network.
      n_residual_blocks: The number of residual blocks to use.
    """

    super().__init__()

    self._input = MaskedConv2d(is_causal=True,
                               in_channels=in_channels,
                               out_channels=2*residual_channels, 
                               kernel_size=7, 
                               padding=3)
    self._masked_layers = nn.ModuleList([
        MaskedResidualBlock(n_channels=2*residual_channels) 
        for _ in range(n_residual_blocks) 
    ])
    self._head = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(in_channels=2*residual_channels, 
                  out_channels=head_channels, 
                  kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=head_channels, 
                  out_channels=in_channels, 
                  kernel_size=1),
        nn.Sigmoid())

  def forward(self, x):
    x = self._input(x)
    skip = torch.zeros_like(x) + x
    for layer in self._masked_layers:
      x = layer(x)
      skip += x
    return self._head(skip)

  def sample(self):
    """Samples a new image.
    
    Args:
      conditioned_on: An (optional) image to condition samples on. Only 
        dimensions with values < 0 will be sampled. For example, if 
        conditioned_on[i] = -1, then output[i] will be sampled conditioned on
        dimensions j < i. If 'None', an unconditional sample will be generated.
    """
    with torch.no_grad():
      device = next(self.parameters()).device
      conditioned_on = (torch.ones((1, 1,  28, 28)) * - 1).to(device)

      for row in range(28):
        for column in range(28):
          for channel in range(1):
            out = self.forward(conditioned_on)[:, channel, row, column]
            out = distributions.Bernoulli(probs=out).sample()
            conditioned_on[:, channel, row, column] = torch.where(
                conditioned_on[:, channel, row, column] < 0,
                out, 
                conditioned_on[:, channel, row, column])
      return conditioned_on

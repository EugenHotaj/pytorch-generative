"""Common modules used by Variational Autoencoders."""

from torch import nn

from pytorch_generative import nn as pg_nn


class ResidualBlock(nn.Module):
    """A simple residual block."""

    def __init__(self, n_channels, hidden_channels):
        """Initializes a new ResidualBlock instance.

        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self._net(x)


class ResidualStack(nn.Module):
    """A stack of multiple ResidualBlocks."""

    def __init__(self, n_channels, hidden_channels, n_residual_blocks=1):
        """Initializes a new ResidualStack instance.

        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
            n_residual_blocks: Number of residual blocks in the stack.
        """
        super().__init__()
        self._net = nn.Sequential(
            *[
                ResidualBlock(n_channels, hidden_channels)
                for _ in range(n_residual_blocks)
            ]
            + [nn.ReLU()]
        )

    def forward(self, x):
        return self._net(x)


class Encoder(nn.Module):
    """A feedforward encoder which downsamples its input."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        n_residual_blocks,
        residual_channels,
        stride,
    ):
        """Initializes a new Encoder instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Number of channels in non residual block hidden layers.
            n_residual_blocks: Number of residual blocks in each residual stack.
            residual_channels: Number of hidden channels in residual blocks.
            stride: Stride to use in the downsampling convolutions. Must be even.
        """
        super().__init__()
        assert stride % 2 == 0, '"stride" must be even.'

        net = []
        for i in range(stride // 2):
            first, last = 0, stride // 2 - 1
            in_c = in_channels if i == first else hidden_channels // 2
            out_c = hidden_channels // 2 if i < last else hidden_channels
            net.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            net.append(nn.ReLU())
        net.append(
            ResidualStack(
                n_channels=hidden_channels,
                hidden_channels=residual_channels,
                n_residual_blocks=n_residual_blocks,
            )
        )
        net.append(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self._net = nn.Sequential(*net)

    def forward(self, x):
        return self._net(x)


class Decoder(nn.Module):
    """A feedforward encoder which upsamples its input."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        n_residual_blocks,
        residual_channels,
        stride,
    ):
        """Initializes a new Decoder instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Number of channels in (non residual block) hidden layers.
            n_residual_blocks: Number of residual blocks in each residual stack.
            residual_channels: Number of hidden channels in residual blocks.
            stride: Stride to use in the upsampling (i.e. transpose) convolutions. Must
                be even.
        """
        super().__init__()

        assert stride % 2 == 0, '"stride" must be even.'

        net = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            ResidualStack(
                n_channels=hidden_channels,
                hidden_channels=residual_channels,
                n_residual_blocks=n_residual_blocks,
            ),
        ]
        for i in range(stride // 2):
            first, last = 0, stride // 2 - 1
            in_c = hidden_channels if i == first else hidden_channels // 2
            out_c = hidden_channels // 2 if i < last else out_channels
            net.append(
                nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if i < last:
                net.append(nn.ReLU())
        self._net = nn.Sequential(*net)

    def forward(self, x):
        return self._net(x)


class Quantizer(nn.Module):
    """Wraps a VectorQuantizer to handle input with arbitrary channels."""

    def __init__(self, in_channels, n_embeddings, embedding_dim):
        """Initializes a new Quantizer instance.

        Args:
            in_channels: Number of input channels.
            n_embeddings: Number of VectorQuantizer embeddings.
            embedding_dim: VectorQuantizer embedding dimension.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=embedding_dim, kernel_size=1
            ),
            pg_nn.VectorQuantizer(n_embeddings, embedding_dim),
        )

    def forward(self, x):
        return self._net(x)

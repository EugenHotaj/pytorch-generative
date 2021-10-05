"""Implementation of emulators of idealized communication channels."""


import torch


class Channel:
    """Base class inherited by all channels in pytorch-generative."""

    def _raise_or_return_valid_prob(self, prob):
        """Raises an Exception if prob is not a valid probability."""
        assert prob >= 0 and prob <= 1.0, f"Invalid probability: '{prob}'"
        return prob

    def transmit(self, message):
        """Emulates transmitting a message over a (possibly noisy) channel.

        Args:
            message: The input message to transmit.
        Returns:
            The output transmitted message. This may not be the same as the input
            message if the channel is noisy.
        """
        pass


class BinarySymmetricChannel(Channel):
    """Implementation of the binary symmetric channel.

    The input message is assumed to be a bitstring. During transmission, each character
    in the bitstring is flipped with probability `prob_flip`.
    """

    def __init__(self, prob_flip):
        """Initializes a new BinarySymmetricChannel instance.

        Args:
            prob_flip: The probability of bit flip during during transmission.
        """
        self.prob_flip = self._raise_or_return_valid_prob(prob_flip)

    def transmit(self, message):
        mask = torch.rand(size=message.shape)
        b_message = message.to(torch.bool)
        result = torch.where(mask < self.prob_flip, ~b_message, b_message)
        return result.to(message.dtype)


class BinaryErasureChannel(Channel):
    """Implementation of the binary erasure (noisy) channel.

    The input message is assumed to be a bitstring. During transmission, each character
    in the bitstring is NaN'ed out with probability `prob_erase`.
    """

    def __init__(self, prob_erase):
        """Initializes a new BinaryErasureChannel instance.

        Args:
            prob_erase: The probability of a bit erasure during transmission.
        """
        self.prob_erase = self._raise_or_return_valid_prob(prob_erase)

    def transmit(self, message):
        mask = torch.rand(size=message.shape)
        nan = torch.tensor(float("nan"))
        return torch.where(mask < self.prob_erase, nan, message)


class BinaryZChannel(Channel):
    """Implementation of the binary z (noisy) channel.

    The input message is assumed to be a bitstring. During transmission, the
    `bit_to_flip` are flipped with with probability `prob_flip`.
    """

    def __init__(self, prob_flip, bit_to_flip=0):
        """Initializes a new BinaryZChannel instance.

        Args:
            prob_flip: The probability of a bit flip during transmission.
            bit_to_flip: Which bit to flip during transmission, 0 or 1.
        """
        self.prob_flip = self._raise_or_return_valid_prob(prob_flip)
        assert bit_to_flip in (0, 1), f"Unknown bit_to_flip: '{bit_to_flip}'."
        self.bit_to_flip = bit_to_flip

    def transmit(self, message):
        mask = torch.rand(size=message.shape)
        b_message = message.to(torch.bool)
        result = torch.where(
            message == self.bit_to_flip,
            torch.where(mask < self.prob_flip, ~b_message, b_message),
            b_message,
        )
        return result.to(message.dtype)

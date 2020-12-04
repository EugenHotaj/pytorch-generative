"""Tests that supported models can call forward() and sample()."""

import tempfile
import unittest

import torch
from torch import distributions

from pytorch_generative import models


class DummyLoader:
    """Dummy data loader used for integration testing."""

    def __init__(self, channels, size):
        self._xs = torch.rand((1, channels, size, size))
        self._ys = torch.tensor([0])

    def __iter__(self):
        self._exhausted = False
        return self

    def __next__(self):
        if not self._exhausted:
            self._exhausted = True
            return self._xs, self._ys
        raise StopIteration()


class IntegrationTests(unittest.TestCase):
    """Main (integration) tests for implemented models."""

    def _test_integration(self, module, in_channels=1, in_size=28):
        dummy_loader = DummyLoader(in_channels, in_size)
        with tempfile.TemporaryDirectory() as log_dir:
            module.reproduce(
                n_epochs=1, log_dir=log_dir, device="cpu", debug_loader=dummy_loader
            )

    # TODO(eugenhotaj): Use parameterized tests.
    def test_NADE(self):
        self._test_integration(models.nade)

    def test_MADE(self):
        self._test_integration(models.made)

    def test_PixelCNN(self):
        self._test_integration(models.pixel_cnn)

    def test_GatedPixelCNN(self):
        self._test_integration(models.gated_pixel_cnn)

    def test_PixelSnail(self):
        self._test_integration(models.pixel_snail)

    def test_ImageGPT(self):
        self._test_integration(models.image_gpt)

    def test_VAE(self):
        self._test_integration(models.vae)

    def test_VeryDeepVAE(self):
        self._test_integration(models.vd_vae, in_size=32)

    def test_VQVAE(self):
        self._test_integration(models.vq_vae, in_channels=3)

    def test_VQVAE2(self):
        self._test_integration(models.vq_vae_2, in_channels=3)


class SmokeTests(unittest.TestCase):
    """Unit tests for things not caught by the integration tests above."""

    def _smoke_test(self, model):
        # Test forward().
        batch = torch.rand(2, 3, 5, 5)
        model(batch)

        # Test unconditional autoregressive sample().
        model.sample(n_samples=2)

        # Test that conditional autoregressive sample() only modifies pixels < 0.
        batch[:, :, 1:, :] = -1
        sample = model.sample(conditioned_on=batch)
        self.assertTrue((sample[:, :, 0, :] == batch[:, :, 0, :]).all())

    def test_PixelCNN_multiple_channels(self):
        model = models.PixelCNN(
            in_channels=3,
            out_channels=3,
            n_residual=1,
            residual_channels=1,
            head_channels=1,
        )
        self._smoke_test(model)

    def test_GatedPixelCNN_multiple_channels(self):
        model = models.GatedPixelCNN(
            in_channels=3, out_channels=3, n_gated=1, gated_channels=1, head_channels=1
        )
        self._smoke_test(model)

    def test_PixelSNAIL_multiple_channels(self):
        model = models.PixelSNAIL(
            in_channels=3,
            out_channels=3,
            n_channels=2,
            n_pixel_snail_blocks=1,
            n_residual_blocks=1,
            attention_key_channels=1,
            attention_value_channels=1,
        )
        self._smoke_test(model)

    def test_ImageGPT_multiple_channels(self):
        model = models.ImageGPT(
            in_channels=3,
            out_channels=3,
            in_size=5,
            n_transformer_blocks=1,
            n_attention_heads=2,
            n_embedding_channels=4,
        )
        self._smoke_test(model)

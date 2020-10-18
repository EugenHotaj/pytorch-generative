"""Tests that supported models can call forward() and sample()."""

import unittest

import torch
from torch import distributions

from pytorch_generative import models


class ModelSmokeTestCase(unittest.TestCase):

  def _smoke_test(self, model, in_channels=1, test_sampling=True):
    shape = (2, in_channels, 5, 5)
    
    # Test forward().
    batch = torch.rand(shape)
    model(batch)

    if test_sampling:
      # Test unconditional sample().
      model.sample(out_shape=shape)

      # Test that conditional sample() only modifies pixels < 0.
      batch[:, :, 1:, :] = -1
      sample = model.sample(conditioned_on=batch)
      self.assertTrue((sample[:, :, 0, :] == batch[:, :, 0, :]).all())

  # TODO(eugenhotaj): Use parameterized tests instead of creating a new method
  # for each model.
  def test_TinyCNN(self):
    model = models.TinyCNN(in_channels=3, out_channels=3)
    self._smoke_test(model, in_channels=3)

  def test_NADE(self):
    model = models.NADE(input_dim=25, hidden_dim=5)
    self._smoke_test(model)

  def test_MADE(self):
    model = models.MADE(input_dim=25, hidden_dims=[32, 32, 32], n_masks=8)
    self._smoke_test(model)

  def test_PixelCNN(self):
    model = models.PixelCNN(in_channels=3,
                            out_channels=3,
                            n_residual=1,
                            residual_channels=1,
                            head_channels=1)
    self._smoke_test(model, in_channels=3)

  def test_GatedPixelCNN(self):
    model = models.GatedPixelCNN(in_channels=3,
                                 out_channels=3,
                                 n_gated=1,
                                 gated_channels=1,
                                 head_channels=1)
    self._smoke_test(model, in_channels=3)

  def test_PixelSNAIL(self):
    model = models.PixelSNAIL(in_channels=3,
                              out_channels=3,
                              n_channels=1,
                              n_pixel_snail_blocks=1,
                              n_residual_blocks=1,
                              attention_key_channels=1,
                              attention_value_channels=1,
                              head_channels=1)
    self._smoke_test(model, in_channels=3) 

  def test_ImageGPT(self):
    model = models.ImageGPT(in_channels=3,
                            in_size=5,
                            out_channels=3,
                            n_transformer_blocks=1,
                            n_attention_heads=2,
                            n_embedding_channels=4)
    self._smoke_test(model, in_channels=3)

  def test_VQVAE(self):
    model = models.VQVAE(in_channels=3,
                         out_channels=3,
                         hidden_channels=2,
                         n_residual_blocks=1,
                         residual_channels=1,
                         n_embeddings=2,
                         embedding_dim=2)
    self._smoke_test(model, in_channels=3, test_sampling=False)


  def test_VQVAE2(self):
    model = models.VQVAE(in_channels=3,
                         out_channels=3,
                         hidden_channels=2,
                         n_residual_blocks=1,
                         residual_channels=1,
                         n_embeddings=2,
                         embedding_dim=2)
    self._smoke_test(model, in_channels=3, test_sampling=False)

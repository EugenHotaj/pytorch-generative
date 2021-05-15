"""Models available in pytorch-generative."""

import torch
from torch import distributions
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base
from pytorch_generative.models.autoregressive.fvbn import FullyVisibleBeliefNetwork
from pytorch_generative.models.autoregressive.gated_pixel_cnn import GatedPixelCNN
from pytorch_generative.models.autoregressive.image_gpt import ImageGPT
from pytorch_generative.models.autoregressive.made import MADE
from pytorch_generative.models.autoregressive.nade import NADE
from pytorch_generative.models.autoregressive.pixel_cnn import PixelCNN
from pytorch_generative.models.autoregressive.pixel_snail import PixelSNAIL
from pytorch_generative.models.kde import GaussianKernel
from pytorch_generative.models.kde import KernelDensityEstimator
from pytorch_generative.models.kde import ParzenWindowKernel
from pytorch_generative.models.mixture_models import BernoulliMixtureModel
from pytorch_generative.models.mixture_models import GaussianMixtureModel
from pytorch_generative.models.vae.beta_vae import BetaVAE
from pytorch_generative.models.vae.vae import VAE
from pytorch_generative.models.vae.vd_vae import VeryDeepVAE
from pytorch_generative.models.vae.vq_vae import VectorQuantizedVAE
from pytorch_generative.models.vae.vq_vae_2 import VectorQuantizedVAE2

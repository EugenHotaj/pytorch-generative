"""Models available in pytorch-generative."""

from pytorch_generative.models.autoregressive.fvbn import FullyVisibleBeliefNetwork
from pytorch_generative.models.autoregressive.gated_pixel_cnn import GatedPixelCNN
from pytorch_generative.models.autoregressive.image_gpt import ImageGPT
from pytorch_generative.models.autoregressive.made import MADE
from pytorch_generative.models.autoregressive.nade import NADE
from pytorch_generative.models.autoregressive.pixel_cnn import PixelCNN
from pytorch_generative.models.autoregressive.pixel_snail import PixelSNAIL
from pytorch_generative.models.flow.nice import NICE
from pytorch_generative.models.kde import (
    GaussianKernel,
    KernelDensityEstimator,
    ParzenWindowKernel,
)
from pytorch_generative.models.mixture_models import (
    BernoulliMixtureModel,
    GaussianMixtureModel,
)
from pytorch_generative.models.vae.beta_vae import BetaVAE
from pytorch_generative.models.vae.vae import VAE
from pytorch_generative.models.vae.vd_vae import VeryDeepVAE
from pytorch_generative.models.vae.vq_vae import VectorQuantizedVAE
from pytorch_generative.models.vae.vq_vae_2 import VectorQuantizedVAE2

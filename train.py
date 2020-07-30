"""Main training script for models."""

import argparse

import torch
from torch import distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

from pytorch_generative import trainer
from pytorch_generative import models


MODEL_MAP = {
    'tiny_cnn': models.TinyCNN,
    'nade': models.NADE,
    'made': models.MADE,
    'gated_pixel_cnn': models.GatedPixelCNN
}


def main(args):
  transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda(lambda x: distributions.Bernoulli(probs=x).sample())
  ])
  train_loader = torch.utils.data.DataLoader(
          datasets.MNIST('./data', train=True, download=True, transform=transform),
          batch_size=args.batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(
          datasets.MNIST('./data', train=False, download=True, transform=transform),
          batch_size=args.batch_size)

  model = MODEL_MAP[args.model](n_channels=1)
  optimizer = optim.Adam(model.parameters())

  criterion = nn.BCELoss(reduction='none')
  def loss_fn(x, _, preds):
    batch_size = x.shape[0]
    x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
    return criterion(preds, x).sum(dim=1).mean()

  model_trainer = trainer.Trainer(model, loss_fn, optimizer, train_loader, 
                                  test_loader, log_dir=args.log_dir)
  model_trainer.interleaved_train_and_eval(n_epochs=args.n_epochs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_dir', type=str, help='the directory where to log data',
      default='/tmp/runs')
  parser.add_argument(
      '--model', type=str, help='the available models to train', 
      default='tiny_cnn', choices=list(MODEL_MAP.keys()))
  parser.add_argument(
      '--batch_size', type=int, help='the training and evaluation batch_size', 
      default=128)
  parser.add_argument(
      '--n_epochs', type=int, help='number of training epochs', default=1)
  args = parser.parse_args()

  main(args)

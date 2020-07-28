"""Simple MNIST model to test the library."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from pytorch_generative.trainer import Trainer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.fc1 = nn.Linear(32*28*28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_epochs', type=int, help='number of training epochs', 
                      default=1)
  args = parser.parse_args()

  transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])
  train_loader = torch.utils.data.DataLoader(
          datasets.MNIST('./data', train=True, download=True, transform=transform),
          batch_size=256, shuffle=True)
  test_loader = torch.utils.data.DataLoader(
          datasets.MNIST('./data', train=False, download=True, transform=transform),
          batch_size=256)

  model = Net()
  optimizer = optim.Adam(model.parameters())
  criterion = nn.NLLLoss()
  loss_fn = lambda x, y, preds: criterion(preds, y)

  trainer = Trainer(model, loss_fn, optimizer, train_loader, test_loader)
  trainer.interleaved_train_and_eval(n_epochs=args.n_epochs)

"""Main training script for models."""	

import argparse	

import torch	
from torch import distributions	
from torch import nn	
from torch import optim	
from torch.optim import lr_scheduler	
from torchvision import datasets	
from torchvision import transforms	

import pytorch_generative as pg	


MODEL_MAP = {	
    'gated_pixel_cnn': pg.models.GatedPixelCNN,	
    'image_gpt': pg.models.ImageGPT,
    'made': pg.models.MADE,	
    'nade': pg.models.NADE,	
    'pixel_cnn': pg.models.PixelCNN,	
    'pixel_snail': pg.models.PixelSNAIL,	
    'vq_vae': pg.models.VQVAE,
    'vq_vae_2': pg.models.VQVAE2,
    'tiny_cnn': pg.models.TinyCNN,	
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

  model = MODEL_MAP[args.model](in_channels=1, out_channels=1)	
  optimizer = optim.Adam(model.parameters())	
  scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda _: 0.9984)	

  criterion = nn.BCEWithLogitsLoss(reduction='none')	
  def loss_fn(x, _, preds):	
    batch_size = x.shape[0]	
    x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))	
    return criterion(preds, x).sum(dim=1).mean()	

  trainer = pg.trainer.Trainer(	
      model, loss_fn, optimizer, train_loader, test_loader, 	
      lr_scheduler=scheduler, log_dir=args.log_dir, save_checkpoint_epochs=1)	
  trainer.interleaved_train_and_eval(n_epochs=args.n_epochs)	


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

"""Utilities to train PyTorch models with less boilerplate."""

import collections

import torch
import tqdm.autonotebook as tqdm

class Trainer:
    """An object which encapsulates the training and evaluation loop.

    Note that the trainer is stateful. This means that calling 
    `trainer.continuous_train_and_eval()` a second time will cause training
    to pick back up from where it left off.
    """

    def __init__(self, model, loss_fn, optimizer, train_loader, eval_loader,
                 device=torch.device('cpu')):
        """Initializes a new Trainer instance.
        
        Args:
            model: The model to train and evaluate.
            loss_fn: A fn(inputs, targets, predictions)->loss.
            optimizer: The optimizer to use when training.
            train_loader: A DataLoader for the training set.
            eval_loader: A DataLoader for the evaluation set.
            device: The device to place the model and data batches on.
        """
        self._model = model.to(device)
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._device = device

        self.train_losses = []
        self.eval_losses = []
        
    # TODO(eugenhotaj): I'm not 100% sure this is the best approach. For 
    # example, to add gradient clipping, we have to override _train_one_batch()
    # just to copy the exact same code plus one extra line. The fastai library
    # uses hooks but that seems like a very heavy-handed approach. Another 
    # option is to just expose gradient_clipping (and other future options) 
    # as __init__ parameters and handle them automatically for the user.
    def _train_one_batch(self, x, y):
        """Trains the model on a single batch of examples.

        Subclasses can override this method to define custom training
        procedures.
        """
        self._optimizer.zero_grad()
        preds = self._model(x)
        loss = self._loss_fn(x, y, preds)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def _eval_one_batch(self, x, y):
        """Evaluates the model on a single batch of examples."""
        preds = self._model(x)
        loss = self._loss_fn(x, y, preds)
        return loss.item()

    def interleaved_train_and_eval(self, n_epochs):
        """Trains and evaluates (after each epoch) for n_epochs."""

        # TODO(eugenhota): Tune this bar formatting. What is useful to log?
        bar_fmt = '{desc}: {percentage:3.0f}% ({rate_fmt}) {postfix}'
 
        for epoch in range(1, n_epochs + 1):
          progress = tqdm.tqdm(
              desc=f'[{epoch}]', unit='batch', bar_format=bar_fmt, 
              total=len(self._train_loader))
          postfix = collections.defaultdict(float)

          # Train.
          self._model.train()
          for x, y, in self._train_loader:
            x, y = x.to(self._device), y.to(self._device)
            train_loss = self._train_one_batch(x, y)
            postfix['train_loss'] = train_loss
            progress.set_postfix(postfix)
            progress.update()

          # Evaluate.
          self._model.eval()
          total_examples, eval_loss = 0, 0.
          with torch.no_grad():
            for x, y, in self._eval_loader:
              x, y = x.to(self._device), y.to(self._device)
              n_examples = x.shape[0]
              total_examples += n_examples
              eval_loss += self._eval_one_batch(x, y) * n_examples
          eval_loss /= total_examples
          postfix['eval_loss'] = eval_loss
          progress.set_postfix(postfix)
          progress.update()
          progress.close()

          # Log.
          self.train_losses.append(train_loss)
          self.eval_losses.append(eval_loss)

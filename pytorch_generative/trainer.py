"""Utilities to train PyTorch models with less boilerplate."""

import time

import torch
from torch.utils import tensorboard

class Trainer:
    """An object which encapsulates the training and evaluation loop.

    Note that the trainer is stateful. This means that calling 
    `trainer.continuous_train_and_eval()` a second time will cause training
    to pick back up from where it left off.
    """

    def __init__(self, 
                 model, 
                 loss_fn, 
                 optimizer, 
                 train_loader, 
                 eval_loader,
                 lr_scheduler=None,
                 log_dir='/tmp/runs',
                 device=torch.device('cpu')):
        """Initializes a new Trainer instance.
        
        Args:
            model: The model to train and evaluate.
            loss_fn: A fn(inputs, targets, predictions)->loss.
            optimizer: The optimizer to use when training.
            train_loader: A DataLoader for the training set.
            eval_loader: A DataLoader for the evaluation set.
            lr_scheduler: A lr_scheduler whose step() method is called after 
              every batch.
            log_dir: The directory where to log TensorBoard metrics.
            device: The device to place the model and data batches on.
        """
        self._model = model.to(device)
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._lr_scheduler = lr_scheduler
        self._log_dir = log_dir
        self._device = device

        self._step = 0
        self._examples_processed = 0
        self._time_taken = 0
        self._summary_writer = tensorboard.SummaryWriter(
            self._log_dir, max_queue=100)

        self.train_losses = []
        self.eval_losses = []
        
    def train_one_batch(self, x, y):
        """Trains the model on a single batch of examples.

        Subclasses can override this method to define custom training loops.
        """
        preds = self._model(x)
        loss = self._loss_fn(x, y, preds)
        return loss

    def _train_one_batch(self, x, y):
      self._model.train()
      x, y = x.to(self._device), y.to(self._device)
      self._optimizer.zero_grad()
      loss = self.train_one_batch(x, y)
      loss.backward()
      self._optimizer.step()
      if self._lr_scheduler is not None:
        self._lr_scheduler.step()
      return loss.item()

    def eval_one_batch(self, x, y):
      """Evaluates the model on a single batch of examples."""
      preds = self._model(x)
      loss = self._loss_fn(x, y, preds)
      return loss

    def _eval_one_batch(self, x, y):
      with torch.no_grad():
        self._model.eval()
        x, y = x.to(self._device), y.to(self._device)
        loss = self.eval_one_batch(x, y)
        return loss.item()

    def interleaved_train_and_eval(self, n_epochs):
        """Trains and evaluates (after each epoch) for n_epochs."""

        for epoch in range(1, n_epochs + 1):
          start_time = time.time()

          # Train.
          train_loss = None
          for i, (x, y), in enumerate(self._train_loader):
            self._examples_processed += x.shape[0]
            lrs = {
                f'group_{i}': param['lr'] 
                for i, param in enumerate(self._optimizer.param_groups)
            }
            self._summary_writer.add_scalars('loss/lr', lrs, self._step)
            train_loss_ = self._train_one_batch(x, y)
            train_loss = (train_loss_ if train_loss is None else 
                          .9 * train_loss + .1 * train_loss_)
            self._summary_writer.add_scalars(
                'loss', {'train': train_loss}, self._step)

            self._time_taken += time.time() - start_time
            start_time = time.time()
            self._summary_writer.add_scalar(
                'perf/examples_per_sec', 
                self._examples_processed / self._time_taken, 
                self._step)
            self._summary_writer.add_scalar(
                'perf/millis_per_example', 
                self._time_taken / self._examples_processed * 1000, 
                self._step)
            self._summary_writer.add_scalar('progress/epoch', epoch, self._step)
            self._summary_writer.add_scalar('progress/step', self._step, self._step)
            self._step += 1
          self.train_losses.append(train_loss)

          # Evaluate
          self._model.eval()
          total_examples, total_loss = 0, 0.
          for x, y, in self._eval_loader:
            n_examples = x.shape[0]
            total_examples += n_examples
            total_loss += self._eval_one_batch(x, y) * n_examples
            eval_loss = total_loss / total_examples
          self._summary_writer.add_scalars('loss', {'eval': eval_loss}, self._step)
          self.eval_losses.append(eval_loss)

          self._summary_writer.flush()

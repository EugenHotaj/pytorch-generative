"""Utilities to train PyTorch models with less boilerplate."""

import os 
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
                 save_checkpoint_epochs=1,
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
            log_dir: The directory where to log checkpoints and TensorBoard 
              metrics.
            save_checkpoint_epochs: The number of epochs to wait before saving
              a new checkpoint. Note that this does not affect TensorBoard 
              logging frequency.
            device: The device to place the model and data batches on.
        """
        # Stateful objects that need to be saved.
        self._model = model.to(device)
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        self._loss_fn = loss_fn
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._log_dir = log_dir
        self._save_checkpoint_epochs = save_checkpoint_epochs
        self._device = device

        self._step = 0
        self._epoch = 0
        self._examples_processed = 0
        self._time_taken = 0

        self._summary_writer = tensorboard.SummaryWriter(
            self._log_dir, max_queue=100)

    def _path(self, file_name):
      return os.path.join(self._log_dir, file_name)

    def _save_checkpoint(self):
      if self._epoch % self._save_checkpoint_epochs != 0:
        return

      torch.save(self._model.state_dict(), self._path('model_state'))
      torch.save(self._optimizer.state_dict(), self._path('optimizer_state'))
      if self._lr_scheduler is not None:
        torch.save(self._lr_scheduler.state_dict(), 
                   self._path('lr_scheduler_state'))
      # TODO(eugenhotaj): Instead of saving these internal counters one at a 
      # time, maybe we can save them as a dictionary.
      torch.save(self._step, self._path('step'))
      torch.save(self._epoch, self._path('epoch'))
      torch.save(self._examples_processed, self._path('examples_processed'))
      torch.save(self._time_taken , self._path('time_taken'))

    def load_from_checkpoint(self):
      """Attempts to load Trainer state from the internal log_dir."""
      self._model.load_state_dict(torch.load(self._path('model_state')))
      self._optimizer.load_state_dict(torch.load(self._path('optimizer_state')))
      if self._lr_scheduler is not None:
        self._lr_scheduler.load_state_dict(
            torch.load(self._path('lr_scheduler_state')))
      self._step = torch.load(self._path('step'))
      self._epoch = torch.load(self._path('epoch'))
      self._examples_processed = torch.load(self._path('examples_processed'))
      self._time_taken = torch.load(self._path('time_taken'))
      # NOTE(eugenhotaj): We need to replace the SummaryWriter and ensure any
      # logs written after the last saved checkpoint are purged.
      self._summary_writer.close()
      self._summary_writer = tensorboard.SummaryWriter(
          self._log_dir, max_queue=100, purge_step=self._step)
      
    def train_one_batch(self, x, y):
      """Trains the model on a single batch of examples.

      Subclasses can override this method to define custom training loops.
      """
      preds = self._model(x)
      loss = self._loss_fn(x, y, preds)
      return loss

    def _train_one_batch(self, x, y):
      self._model.train()
      x = x.to(self._device)
      if y is not None:
        y = y.to(self._device)
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
        x = x.to(self._device)
        if y is not None:
          y = y.to(self._device)
        loss = self.eval_one_batch(x, y)
        return loss.item()

    def interleaved_train_and_eval(self, n_epochs):
      """Trains and evaluates (after each epoch) for n_epochs."""

      for _ in range(n_epochs):
        start_time = time.time()

        # Train.
        for i, batch in enumerate(self._train_loader):
          batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
          x, y = batch
          self._examples_processed += x.shape[0]
          lrs = {
              f'group_{i}': param['lr'] 
              for i, param in enumerate(self._optimizer.param_groups)
          }
          self._summary_writer.add_scalars('loss/lr', lrs, self._step)
          train_loss = self._train_one_batch(x, y)
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
          self._summary_writer.add_scalar('progress/epoch', self._epoch, self._step)
          self._summary_writer.add_scalar('progress/step', self._step, self._step)
          self._step += 1

        # Evaluate
        self._model.eval()
        total_examples, total_loss = 0, 0.
        for batch in self._eval_loader:
          batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
          x, y = batch
          n_examples = x.shape[0]
          total_examples += n_examples
          total_loss += self._eval_one_batch(x, y) * n_examples
          eval_loss = total_loss / total_examples
        self._summary_writer.add_scalars('loss', {'eval': eval_loss}, self._step)

        self._epoch += 1
        self._save_checkpoint()
      self._summary_writer.close()

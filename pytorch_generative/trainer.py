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
        self._log_dir = log_dir
        self._device = device

        self._summary_writer = tensorboard.SummaryWriter(
            self._log_dir, max_queue=100)

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
        return loss

    def _eval_one_batch(self, x, y):
        """Evaluates the model on a single batch of examples."""
        preds = self._model(x)
        loss = self._loss_fn(x, y, preds)
        return loss

    def interleaved_train_and_eval(self, n_epochs):
        """Trains and evaluates (after each epoch) for n_epochs."""

        step, examples_processed, time_taken = 0, 0, 0.,
        for epoch in range(1, n_epochs + 1):
          start_time = time.time()

          # Train.
          self._model.train()
          train_loss = None
          for i, (x, y), in enumerate(self._train_loader):
            x, y = x.to(self._device), y.to(self._device)
            examples_processed += x.shape[0]
            if train_loss is None:
              train_loss =  self._train_one_batch(x, y).item()
            else:
              train_loss = .9 * train_loss + .1 * self._train_one_batch(x, y).item()
            self._summary_writer.add_scalars('loss', 
                                             {'train': train_loss}, step)

            time_taken += time.time() - start_time
            start_time = time.time()
            self._summary_writer.add_scalar(
                'perf/examples_per_sec', examples_processed/time_taken, step)
            self._summary_writer.add_scalar(
                'perf/millis_per_example', 
                time_taken/examples_processed * 1000, 
                step)
            self._summary_writer.add_scalar('progress/epoch', epoch, step)
            self._summary_writer.add_scalar('progress/step', step, step)
            step += 1
          self.train_losses.append(train_loss)

          # Evaluate
          self._model.eval()
          total_examples, total_loss = 0, 0.
          with torch.no_grad():
            for x, y, in self._eval_loader:
              x, y = x.to(self._device), y.to(self._device)
              n_examples = x.shape[0]
              total_examples += n_examples
              total_loss += self._eval_one_batch(x, y).item() * n_examples
              eval_loss = total_loss / total_examples
          self._summary_writer.add_scalars('loss', {'eval': eval_loss}, step)
          self.eval_losses.append(eval_loss)

          self._summary_writer.flush()

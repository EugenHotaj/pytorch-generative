"""Utilities to train PyTorch models with less boilerplate."""

import torch
import time


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
        
    def _run_one_epoch(self, batch_fn, loader):
        """Runs and instruments one epoch.

        Args:
            batch_fn: The fn(Xs, ys)->loss which takes in a batch of data and 
                returns a scalar loss. The batch_fn can do whatever it wants, 
                such as train a model on the batch, eval on the batch, etc. 
            loader: The DataLoader to use.
        """
        start_time = time.time()
        loss, n_examples = 0., 0
        for x, y, in loader:
            x, y = x.to(self._device), y.to(self._device)
            batch_size = x.shape[0]
            n_examples += batch_size
            batch_loss = batch_fn(x, y)
            loss += batch_loss.item() * batch_size
        total_time = time.time() - start_time
        examples_per_sec = round(n_examples / total_time)
        loss /= n_examples
        return loss, examples_per_sec

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
        return self._loss_fn(x, y, preds)

    def interleaved_train_and_eval(self, n_epochs):
        """Trains and evaluates (after each epoch) for n_epochs."""
        for _ in range(1, n_epochs + 1):
            # Train.
            self._model.train()
            train_loss, examples_per_sec = self._run_one_epoch(
                    self._train_one_batch, self._train_loader)

            # Evaluate.
            self._model.eval()
            with torch.no_grad():
                eval_loss, _ = self._run_one_epoch(
                        self._eval_one_batch, self._eval_loader)

            # Log.
            self.train_losses.append(train_loss)
            self.eval_losses.append(eval_loss)
            print(f'[{len(self.train_losses)}|{examples_per_sec}]: '
                  f'train_loss={train_loss} eval_loss={eval_loss}')

"""Utilities to train PyTorch models with less boilerplate."""

import collections
import glob
import os
import re
import tempfile
import time

import torch
from torch.nn import utils
from torch.utils import tensorboard


class Trainer:
    """An object which encapsulates the training and evaluation loop.

    Note that the trainer is stateful. This means that calling
    `trainer.continuous_train_and_eval()` a second time will cause training
    to pick back up from where it left off.
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        eval_loader,
        lr_scheduler=None,
        clip_grad_norm=None,
        skip_grad_norm=None,
        sample_epochs=None,
        sample_fn=None,
        log_dir=None,
        save_checkpoint_epochs=1,
        device="cpu",
    ):
        """Initializes a new Trainer instance.

        Args:
            model: Model to train and evaluate.
            loss_fn: A `fn(inputs, targets, predictions)->output`. The output can either
                be a single loss Tensor or a dictionary containing multiple loss
                Tensors. The dictionary must contain a `loss` key which will be used as
                the primary loss for backprop.
            optimizer: Optimizer to use when training.
            train_loader: DataLoader for the training set.
            eval_loader: DataLoader for the evaluation set.
            lr_scheduler: An torch.optim.lr_scheduler whose step() method is called
                after every batch.
            clip_grad_norm: L2 norm to scale gradients to if their norm is greater.
            skip_grad_norm: Maximum L2 norm above which gradients are discarded.
            sample_epochs: Number of epochs to wait between generating new image samples
                and logging them to TensorBoard. If not `None`, `sample_fn` must be
                provided.
            sample_fn: A `fn(model)->Tensor` which returns an NCHW Tensor of images to
                log to TensorBoard.
            log_dir: The directory where to log checkpoints and TensorBoard metrics. If
                `None` a temporary directory is created (note that this directory is not
                cleaned up automatically).
            save_checkpoint_epochs: Number of epochs to wait between checkpoints. Note
                that this does not affect TensorBoard logging frequency.
            device: The device to place the model and data. Either string or
                torch.device.
        """
        # Stateful objects that need to be saved.
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.clip_grad_norm = clip_grad_norm
        self.skip_grad_norm = skip_grad_norm
        self.log_dir = log_dir or tempfile.mkdtemp()
        self.save_checkpoint_epochs = save_checkpoint_epochs
        self.device = torch.device(device) if isinstance(device, str) else device

        self.sample_epochs = sample_epochs
        self.sample_fn = sample_fn
        if self.sample_epochs:
            msg = "sample_fn cannot be None if sample_epochs is not None"
            assert self.sample_fn, msg

        self._step = 0
        self._epoch = 0
        self._examples_processed = 0
        self._time_taken = 0

        self._summary_writer = tensorboard.SummaryWriter(self.log_dir, max_queue=100)

    def _path(self, file_name):
        return os.path.join(self.log_dir, file_name)

    def _save_checkpoint(self):
        if self._epoch % self.save_checkpoint_epochs != 0:
            return
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "epoch": self._epoch,
            "examples_processed": self._examples_processed,
            "time_taken": self._time_taken,
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
        # TODO(eugenhotaj): Add an option to keep only the last n checkpoints.
        torch.save(checkpoint, self._path(f"trainer_state_{self._epoch}.ckpt"))

    def _find_latest_epoch(self):
        files = glob.glob(self._path("trainer_state_[0-9]*.ckpt"))
        epochs = sorted([int(re.findall(r"\d+", f)[0]) for f in files])
        if not epochs:
            raise FileNotFoundError(f"No checkpoints found in {self.log_dir}.")
        return epochs[-1]

    def restore_checkpoint(self, epoch=None):
        """Restores the Trainer's state using self.log_dir.

        Args:
            epoch: Epoch from which to restore the Trainer's state. If None, uses the
                latest available epoch.
        """
        epoch = epoch or self._find_latest_epoch()
        checkpoint = torch.load(self._path(f"trainer_state_{epoch}.ckpt"))

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step = checkpoint["step"]
        self._epoch = checkpoint["epoch"]
        self._examples_processed = checkpoint["examples_processed"]
        self._time_taken = checkpoint["time_taken"]
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # NOTE(eugenhotaj): We need to replace the SummaryWriter and ensure any
        # logs written after the last saved checkpoint are purged.
        self._summary_writer.close()
        self._summary_writer = tensorboard.SummaryWriter(
            self.log_dir, max_queue=100, purge_step=self._step
        )

    def _get_loss_dict(self, loss):
        loss = loss if isinstance(loss, dict) else {"loss": loss}
        assert "loss" in loss, 'Losses dictionary does not contain "loss" key.'
        return loss

    # TODO(eugenhotaj): Consider removing the 'training' argument and just using
    # self.model.parameters().training.
    def _log_loss_dict(self, loss_dict, training):
        for key, loss in loss_dict.items():
            key = key if key == "loss" else f"loss/{key}"
            self._summary_writer.add_scalars(
                key, {"train" if training else "eval": loss}, self._step
            )

    def train_one_batch(self, x, y):
        """Trains the model on a single batch of examples.

        Subclasses can override this method to define custom training loops.
        """
        preds = self.model(x)
        loss = self.loss_fn(x, y, preds)
        return loss

    def _train_one_batch(self, x, y):
        self.model.train()
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
        self.optimizer.zero_grad()
        loss = self._get_loss_dict(self.train_one_batch(x, y))
        loss["loss"].backward()

        norm = 0
        max_norm = self.clip_grad_norm or self.skip_grad_norm or None
        if max_norm:
            norm = utils.clip_grad_norm(self.model.parameters(), max_norm).item()

        if not self.skip_grad_norm or norm <= self.skip_grad_norm:
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return {k: v.item() for k, v in loss.items()}

    def eval_one_batch(self, x, y):
        """Evaluates the model on a single batch of examples.

        Subclasses can override this method to define custom evaluation loops.
        """
        preds = self.model(x)
        loss = self.loss_fn(x, y, preds)
        return loss

    def _eval_one_batch(self, x, y):
        with torch.no_grad():
            self.model.eval()
            x = x.to(self.device)
            if y is not None:
                y = y.to(self.device)
            loss = self._get_loss_dict(self.eval_one_batch(x, y))
            return {k: v.item() for k, v in loss.items()}

    def interleaved_train_and_eval(self, max_epochs, restore=True):
        """Trains and evaluates (after each epoch).

        Args:
            max_epochs: Maximum number of epochs to train for.
            restore: Wether to continue training from an existing checkpoint in
                self.log_dir.
        """
        if restore:
            try:
                self.restore_checkpoint()
            except FileNotFoundError:
                pass  # No checkpoint found in self.log_dir; train from scratch.

        for _ in range(max_epochs - self._epoch):
            start_time = time.time()

            # Train.
            for i, batch in enumerate(self.train_loader):
                batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
                x, y = batch
                self._examples_processed += x.shape[0]
                lrs = {
                    f"group_{i}": param["lr"]
                    for i, param in enumerate(self.optimizer.param_groups)
                }
                self._summary_writer.add_scalars("loss/lr", lrs, self._step)
                loss = self._train_one_batch(x, y)
                self._log_loss_dict(loss, training=True)

                self._time_taken += time.time() - start_time
                start_time = time.time()
                self._summary_writer.add_scalar(
                    "speed/examples_per_sec",
                    self._examples_processed / self._time_taken,
                    self._step,
                )
                self._summary_writer.add_scalar(
                    "speed/millis_per_example",
                    self._time_taken / self._examples_processed * 1000,
                    self._step,
                )
                self._summary_writer.add_scalar("speed/epoch", self._epoch, self._step)
                self._summary_writer.add_scalar("speed/step", self._step, self._step)
                self._step += 1

            # Evaluate
            total_examples, total_loss = 0, collections.defaultdict(int)
            for batch in self.eval_loader:
                batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
                x, y = batch
                n_examples = x.shape[0]
                total_examples += n_examples
                for key, loss in self._eval_one_batch(x, y).items():
                    total_loss[key] += loss * n_examples
            loss = {key: loss / total_examples for key, loss in total_loss.items()}
            self._log_loss_dict(loss, training=False)

            self._epoch += 1
            self._save_checkpoint()
            if self.sample_epochs and self._epoch % self.sample_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    tensor = self.sample_fn(self.model)
                self._summary_writer.add_images("sample", tensor, self._step)

        self._summary_writer.close()

"""Extra generative modeling benchmark datasets not provided by PyTorch."""

import os
import urllib

import PIL
import numpy as np
import torch
from torch import distributions
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import utils
from torchvision.datasets import vision


def _dynamically_binarize(x):
    return distributions.Bernoulli(probs=x).sample()


def _resize_to_32(x):
    return F.pad(x, (2, 2, 2, 2))


def get_mnist_loaders(batch_size, dynamically_binarize=False, resize_to_32=False):
    """Create train and test loaders for the MNIST dataset.

    Args:
        batch_size: The batch size to use.
        dynamically_binarize: Whether to dynamically  binarize images values to {0, 1}.
        resize_to_32: Whether to resize the images to 32x32.
    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform = [transforms.ToTensor()]
    if dynamically_binarize:
        transform.append(_dynamically_binarize)
    if resize_to_32:
        transform.append(_resize_to_32)
    transform = transforms.Compose(transform)
    train_loader = data.DataLoader(
        datasets.MNIST("/tmp/data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    test_loader = data.DataLoader(
        datasets.MNIST("/tmp/data", train=False, download=True, transform=transform),
        batch_size=batch_size,
        num_workers=os.cpu_count(),
    )
    return train_loader, test_loader


def get_cifar10_loaders(batch_size, normalize=False):
    """Create train and test loaders for the CIFAR10 dataset.

    Args:
        batch_size: The batch size to use.
        normalize: Whether to normalize images to be zero mean, unit variance.
    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform = [transforms.ToTensor()]
    if normalize:
        transform.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
    transform = transforms.Compose(transform)
    train_loader = data.DataLoader(
        datasets.CIFAR10("/tmp/data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    test_loader = data.DataLoader(
        datasets.CIFAR10("/tmp/data", train=False, download=True, transform=transform),
        batch_size=batch_size,
        num_workers=os.cpu_count(),
    )
    return train_loader, test_loader


def _read_image_file(path, shape):
    with open(path, "rb") as f:
        images = np.loadtxt(f, delimiter=" ", dtype=np.uint8) * 255
    return torch.from_numpy(images).view(-1, *shape)


class BinarizedMNIST(vision.VisionDataset):
    """A specific binarization of the MNIST images.

    Originally used in Salakhutdinov & Murray (2008). This dataset is used to evaluate
    generative models of images, so labels are not provided.

    NOTE: The evaluation split is merged into the training set.
    """

    _URL = (
        "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
        "binarized_mnist_"
    )
    resources = [_URL + "train.amat", _URL + "valid.amat", _URL + "test.amat"]
    train_file = "train.pt"
    valid_file = "valid.pt"
    test_file = "test.pt"

    def __init__(self, root, split="train", transform=None):
        """Initializes a new BinarizedMNIST instance.

        Args:
            root: The directory containing the data. If the data does not exist, it will
                be download to this directory.
            split: Which split to use. Must be one of 'train', 'valid', or 'test'.
            transform: A torchvision.transform to apply to the data.
        """
        super().__init__(root, transform=transform)
        assert split in ("train", "valid", "test")
        self._raw_folder = os.path.join(self.root, "BinarizedMNIST", "raw")
        self._folder = os.path.join(self.root, "BinarizedMNIST")
        self.train = train
        if not self._check_exists():
            self.download()
        self.data = torch.load(os.path.join(self._folder, split + ".pt"))

    def __getitem__(self, index):
        """Returns the tuple (img, None) with the given index."""
        img = self.data[index]
        # Return PIL images to be connsistent with other datasets.
        img = PIL.Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self._folder, self.train_file)
        ) and os.path.exists(os.path.join(self._folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in the root folder."""
        if self._check_exists():
            return

        # Download files.
        os.makedirs(self._folder, exist_ok=True)
        os.makedirs(self._raw_folder, exist_ok=True)
        for url in self.resources:
            filename = url.rpartition("/")[-1]
            utils.download_url(url, root=self._raw_folder, filename=filename)

        # Process and save.
        shape = 28, 28
        train_set = _read_image_file(
            os.path.join(self._raw_folder, "binarized_mnist_train.amat"), shape
        )
        with open(os.path.join(self._folder, self.train_file), "wb") as f:
            torch.save(train_set, f)
        valid_set = _read_image_file(
            os.path.join(self._raw_folder, "binarized_mnist_valid.amat"), shape
        )
        with open(os.path.join(self._folder, self.valid_file), "wb") as f:
            torch.save(valid_set, f)
        test_set = _read_image_file(
            os.path.join(self._raw_folder, "binarized_mnist_test.amat"), shape
        )
        with open(os.path.join(self._folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train else "Test")

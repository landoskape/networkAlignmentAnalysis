import multiprocessing
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2 as transforms

from . import files
from .models.base import AlignmentNetwork

REQUIRED_PROPERTIES = ["dataset_path", "dataset_constructor", "loss_function"]


def default_loader_parameters(
    distributed,
    batch_size=1024,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
):
    """
    contains the default dataloader parameters with the option of updating them
    using key word argument
    """
    default_parameters = dict(
        batch_size=batch_size,
        num_workers=num_workers,  # usually 2 workers is appropriate for swapping loading during batch processing
        shuffle=False if distributed else shuffle,  # can't use shuffle=True if using DDP
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return default_parameters


class DataSet(ABC):
    def __init__(
        self,
        device=None,
        distributed=False,
        dataset_parameters={},
        transform_parameters={},
        loader_parameters={},
    ):
        # set properties of dataset and check that all required properties are defined
        self.set_properties()
        self.check_properties()

        # define device for dataloading
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.distributed = distributed

        # define extra transform (should be a callable method or None) for any transformations that
        # can't go in the torchvision.transforms.Compose(...), hopefully this won't be needed later
        # when that issue is resolved (grayscale to RGB transform isn't working in Compose right now)
        self.extra_transform = transform_parameters.pop("extra_transform", None)

        # create transform for dataloader
        self.transform_parameters = transform_parameters
        self.make_transform(**transform_parameters)

        # define the dataloader parameters
        self.dataloader_parameters = default_loader_parameters(distributed, **loader_parameters)  # get dataloader parameters

        # load the dataset and create the dataloaders
        self.dataset_parameters = dataset_parameters
        self.load_dataset(**dataset_parameters)

    def check_properties(self):
        """
        DataSet objects have a few properties that are required but need to be set
        by the children. For simplicity, I want the properties to be attributes,
        instead of @property methods, but I want to make sure that all the required
        properties are loaded. Hence this method.
        """
        if not all([hasattr(self, prop) for prop in REQUIRED_PROPERTIES]):
            not_found = [prop for prop in REQUIRED_PROPERTIES if not hasattr(self, prop)]
            raise ValueError(f"The following required properties were not set: {not_found}")

    @abstractmethod
    def set_properties(self):
        """
        DataSets have a few required properties (listed in the **REQUIRED_PROPERTIES**
        global variable). This method is used to set them all. There is a check implemented
        in the __init__ method to make sure all required properties are set.

        required
        --------
        dataset_path: string, defines the local file location containing the relevant dataset
                      files. in particular, whatever filepath is returned by this method will
                      be passed into the "root" input for torch datasets that is required to
                      load one of the standard datasets (like MNIST, CIFAR, ImageNet...)

        dataset_constructor: callable method, defines the constructor object for the dataset
        loss_function: callable method, defines how to evaluate the loss of the output and target
        """
        pass

    @abstractmethod
    def dataset_kwargs(self, train=True, **kwargs):
        """
        keyword arguments passed into the torch dataset constructor

        different datasets have different kwarg requirements, including
        the way train vs test is defined by the kwargs. This class calls
        the train dataset and the test dataset using dataset_kwargs(train=True)
        or train=False, so the children need to define whatever kwargs go
        into the dataset_kwargs appropriately to be determined by the kwarg
        train
        """
        pass

    def load_dataset(self, **kwargs):
        """load dataset using the established path and parameters"""
        self.train_dataset = self.dataset_constructor(**self.dataset_kwargs(train=True, **kwargs))
        self.test_dataset = self.dataset_constructor(**self.dataset_kwargs(train=False, **kwargs))
        self.train_sampler = DistributedSampler(self.train_dataset) if self.distributed else None
        self.test_sampler = DistributedSampler(self.test_dataset) if self.distributed else None
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=self.train_sampler, **self.dataloader_parameters)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, sampler=self.test_sampler, **self.dataloader_parameters)

    def unwrap_batch(self, batch, device=None):
        """simple method for unwrapping batch for simple training loops"""
        device = self.device if device is None else device
        if self.extra_transform:
            batch = self.extra_transform(batch)
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        return inputs, targets

    def make_transform(self, center_crop=None, resize=None, flatten=False, out_channels=None):
        """
        create transform for dataloader
        resize is the new (H, W) shape of the image for the transforms.Resize transform (or None)
        flatten is a boolean indicating whether to flatten the image, (i.e. for a linear input layer)
        """
        # default transforms
        use_transforms = [
            # Convert PIL Image to PyTorch Tensor
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]

        if center_crop:
            use_transforms.append(transforms.CenterCrop(center_crop))

        # Normalize inputs to canonical distribution
        use_transforms.append(transforms.Normalize((self.dist_params["mean"]), (self.dist_params["std"])))

        # extra transforms depending on network
        if resize:
            use_transforms.append(transforms.Resize(resize, antialias=True))
        if out_channels:
            use_transforms.append(transforms.Grayscale(num_output_channels=out_channels))
        if flatten:
            use_transforms.append(transforms.Lambda(torch.flatten))

        # store composed transformation
        self.transform = transforms.Compose(use_transforms)

    def measure_loss(self, outputs, targets, reduction=None):
        """simple method for measuring loss with stored loss function"""
        if reduction is None:
            return self.loss_function(outputs, targets)

        standard_reduction = self.loss_function.reduction
        self.loss_function.reduction = reduction
        loss = self.loss_function(outputs, targets)
        self.loss_function.reduction = standard_reduction
        return loss

    def measure_accuracy(self, outputs, targets, k=1, percentage=True):
        """
        simple method for measuring accuracy on a classification problem

        default output is top1 percentage, but k can set the top-k accuracy
        and if percentage=False then returns the number correct (by topk)
        """
        topk = outputs.topk(k, dim=1, sorted=True, largest=True)[1]  # get topk indices
        num_correct = torch.sum(torch.any(topk == targets.view(-1, 1), dim=1))  # num correct
        if percentage:
            return 100 * num_correct / outputs.size(0)  # percentage
        else:
            return num_correct


class MNIST(DataSet):
    def set_properties(self):
        """defines the required properties for MNIST"""
        self.dataset_path = files.dataset_path("MNIST")
        self.dataset_constructor = torchvision.datasets.MNIST
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.1307], std=[0.3081])

    def dataset_kwargs(self, train=True, download=False):
        """set data constructor kwargs for MNIST"""
        kwargs = dict(
            train=train,
            root=self.dataset_path,
            download=download,
            transform=self.transform,
        )
        return kwargs


class CIFAR10(DataSet):
    def set_properties(self):
        """defines the required properties for CIFAR10"""
        self.dataset_path = files.dataset_path("CIFAR10")
        self.dataset_constructor = torchvision.datasets.CIFAR10
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def dataset_kwargs(self, train=True, download=False):
        """set data constructor kwargs for CIFAR10"""
        kwargs = dict(
            train=train,
            root=self.dataset_path,
            download=download,
            transform=self.transform,
        )
        return kwargs


class CIFAR100(CIFAR10):
    def set_properties(self):
        """defines the required properties for CIFAR100"""
        self.dataset_path = files.dataset_path("CIFAR100")
        self.dataset_constructor = torchvision.datasets.CIFAR100
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


class ImageNet2012(DataSet):
    def set_properties(self):
        """
        defines the required properties for ImageNet 2012 (ILSVRC2012) with
        1000 classes.
        preprocessing according to pytorch documentation:
        https://pytorch.org/hub/pytorch_vision_alexnet/
        """
        self.dataset_path = files.dataset_path("ImageNet")
        self.dataset_constructor = torchvision.datasets.ImageNet
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.center_crop = 224

    def dataset_kwargs(self, train=True):
        """set data constructor kwargs for ImageNet2012"""
        kwargs = dict(
            split="train" if train else "val",
            root=self.dataset_path,
            transform=self.transform,
        )
        return kwargs


DATASET_REGISTRY = {
    "MNIST": MNIST,
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "ImageNet": ImageNet2012,
}


def get_dataset(
    dataset_name,
    build=False,
    dataset_parameters={},
    transform_parameters={},
    loader_parameters={},
    **kwargs,
):
    """
    lookup dataset constructor from dataset registry by name

    if build=True, uses kwargs to build dataset and returns a dataset object
    otherwise just returns the constructor
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset ({dataset_name}) is not in DATASET_REGISTRY")
    dataset = DATASET_REGISTRY[dataset_name]
    if build:
        if isinstance(transform_parameters, AlignmentNetwork):
            # Can use an AlignmentNetwork instance to automatically retrieve transform parameters
            transform_parameters = transform_parameters.get_transform_parameters(dataset_name)
        else:
            if not isinstance(transform_parameters, dict):
                raise TypeError("transform_parameters must be a dictionary or an AlignmentNetwork")

        # Build the dataset
        return dataset(
            dataset_parameters=dataset_parameters,
            transform_parameters=transform_parameters,
            loader_parameters=loader_parameters,
            **kwargs,
        )

    # Otherwise return the constructor
    return dataset


if __name__ == "__main__":
    """simple program for downloading a dataset"""

    from argparse import ArgumentParser

    def get_args(args=None):
        parser = ArgumentParser(description="simple program for downloading a dataset to the local file location")
        parser.add_argument("--dataset", type=str, default="MNIST")
        return parser.parse_args(args=args)

    args = get_args()

    dataset = get_dataset(args.dataset, build=True, dataset_parameters=dict(download=True))

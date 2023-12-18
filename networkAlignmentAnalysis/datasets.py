from abc import ABC, abstractmethod 
import torch
from torch import nn
import torchvision 
from torchvision import transforms
import multiprocessing

from . import files


REQUIRED_PROPERTIES = ['dataset_path', 'dataset_constructor', 'loss_function']

def default_loader_parameters(batch_size=1024):
    default_parameters = dict(
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count()-2, # use the computer without stealing all resources
        shuffle=True,
    )
    return default_parameters

class DataSet(ABC):
    def __init__(self, device=None, transform_parameters={}, loader_parameters={}):
        # set properties of dataset and check that all required properties are defined
        self.set_properties() 
        self.check_properties() 

        # define device for dataloading
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        # define extra transform (should be a callable method or None) for any transformations that 
        # can't go in the torchvision.transforms.Compose(...), hopefully this won't be needed later 
        # when that issue is resolved (grayscale to RGB transform isn't working in Compose right now)
        self.extra_transform = transform_parameters.pop('extra_transform', None)

        # create transform for dataloader
        self.transform_parameters = transform_parameters 
        self.make_transform(**transform_parameters) 

        # define the dataloader parameters
        self.dataloader_parameters = default_loader_parameters(**loader_parameters) # get dataloader parameters
        
        # load the dataset and create the dataloaders
        self.load_dataset() 

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
    def make_transform(self, **transform_parameters):
        """
        defines the relevant transforms in the ETL pipeline for the dataset
        
        requires kwargs "transform_parameters" that are provided at initialization, 
        stored by the object, and automatically passed into this method. It is
        structured like this because I've unpacked the kwargs in the make_transform
        methods to have clear requirements and provide defaults
        """
        pass
    
    @abstractmethod
    def dataset_kwargs(self, train=True):
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

    def load_dataset(self):
        """load dataset using the established path and parameters"""
        self.train_dataset = self.dataset_constructor(**self.dataset_kwargs(train=True))
        self.test_dataset = self.dataset_constructor(**self.dataset_kwargs(train=False))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, **self.dataloader_parameters)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, **self.dataloader_parameters)

    def unwrap_batch(self, batch, device=None):
        """simple method for unwrapping batch for simple training loops"""
        device = self.device if device is None else device
        if self.extra_transform: batch = self.extra_transform(batch)
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        return inputs, targets
    
    def measure_loss(self, outputs, targets):
        """simple method for measuring loss with stored loss function"""
        return self.loss_function(outputs, targets)
    
    @abstractmethod
    def measure_performance(self, outputs, targets):
        """simple method for measuring performance with any metrics other than the loss"""
        pass


class MNIST(DataSet):
    def set_properties(self):
        """defines the required properties for MNIST"""
        self.dataset_path = files.dataset_path("MNIST")
        self.dataset_constructor = torchvision.datasets.MNIST
        self.loss_function = nn.CrossEntropyLoss()

    def make_transform(self, resize=None, flatten=False):
        """
        create transform for dataloader
        resize is the new (H, W) shape of the image for the transforms.Resize transform (or None)
        flatten is a boolean indicating whether to flatten the image, (i.e. for a linear input layer)
        """
        # default transforms
        use_transforms = [
            transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
            transforms.Normalize((0.1307,), (0.3081,)), # Normalize inputs to canonical distribution
            ]
        
        # extra transforms depending on network
        if resize:
            use_transforms.append(transforms.Resize(resize))
        if flatten:
            use_transforms.append(transforms.Lambda(torch.flatten))

        # store composed transformation
        self.transform = transforms.Compose(use_transforms)

    def dataset_kwargs(self, train=True):
        """set data constructor kwargs for MNIST"""
        kwargs = dict(
            train=train,
            root=self.dataset_path,
            download=True,
            transform=self.transform,
        )
        return kwargs
    
    def measure_performance(self, outputs, targets, k=1, percentage=True):
        """performance on mnist measure by top1 accuracy"""
        topk = outputs.topk(k, dim=1, sorted=True, largest=True)[1]
        out = torch.sum(torch.any(topk==targets.view(-1, 1), dim=1)) # num correct
        if percentage: 
            out = 100 * out/outputs.size(0) # percentage
        return out


DATASET_REGISTRY = {
    'MNIST': MNIST,
}

def get_dataset(dataset_name):
    """lookup dataset constructor from dataset registry by name"""
    if dataset_name not in DATASET_REGISTRY: 
        raise ValueError(f"Dataset ({dataset_name}) is not in DATASET_REGISTRY")
    return DATASET_REGISTRY[dataset_name]

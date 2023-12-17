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
    def __init__(self, transform_parameters={}, loader_parameters={}):
        self.set_properties() # set properties
        self.check_properties() # check if all required properties were set correctly
        self.transform_parameters = transform_parameters # save transform parameters for good prudence
        self.make_transform(**transform_parameters) # make torch transform for dataloader
        self.dataloader_parameters = default_loader_parameters(**loader_parameters) # get dataloader parameters
        self.load_dataset() # load datasets and create dataloaders

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

    def unwrap_batch(self, batch, device='cpu'):
        """simple method for unwrapping batch for simple training loops"""
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        return images, labels
    
    def measure_loss(self, outputs, labels):
        """simple method for measuring loss with stored loss function"""
        return self.loss_function(outputs, labels)


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
        resize = transforms.Resize(resize) if resize is not None else None
        flatten = transforms.Lambda(torch.flatten) if flatten else None
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
            transforms.Normalize((0.1307,), (0.3081,)), # Normalize inputs to canonical distribution
            resize, 
            flatten,
        ])

    def dataset_kwargs(self, train=True):
        """set data constructor kwargs for MNIST"""
        kwargs = dict(
            train=train,
            root=self.dataset_path,
            download=True,
            transform=self.transform,
        )
        return kwargs


"""
Should turn these into classes for loading...
Also I want the measure performance method to be a part of a standard dataloader class
"""

def measurePerformance(net, dataloader, DEVICE=None, verbose=False):
    if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Measure performance
    loss_function = nn.CrossEntropyLoss()
    totalLoss = 0
    numCorrect = 0
    numAttempted = 0
    
    if verbose: iterator = tqdm(dataloader)
    else: iterator = dataloader
    
    for batch in iterator:
        images, label = batch
        images = images.to(DEVICE)
        label = label.to(DEVICE)
        outputs = net(images)
        totalLoss += loss_function(outputs,label).item()
        output1 = torch.argmax(outputs,axis=1)
        numCorrect += sum(output1==label)
        numAttempted += images.shape[0]
        
    return totalLoss/len(dataloader), 100*numCorrect/numAttempted




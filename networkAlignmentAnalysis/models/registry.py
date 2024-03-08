from . import models

MODEL_REGISTRY = {
    "MLP": models.MLP,
    "CNN2P2": models.CNN2P2,
    "AlexNet": models.AlexNet,
}

DATASET_ARGUMENTS = {
    "MLP": {
        "MNIST": dict(input_dim=784, output_dim=10),
        "CIFAR10": dict(input_dim=3072, output_dim=10),
        "CIFAR100": dict(input_dim=3072, output_dim=100),
    },
    "CNN2P2": {
        "MNIST": dict(in_channels=1, output_dim=10),
        "CIFAR10": dict(in_channels=3, num_hidden=[4096, 128], output_dim=10),
        "CIFAR100": dict(in_channels=3, num_hidden=[4096, 128], output_dim=100),
        "ImageNet": dict(in_channels=3, output_dim=1000),
    },
    "AlexNet": {
        "MNIST": dict(num_classes=10),
        "CIFAR10": dict(num_classes=10),
        "CIFAR100": dict(num_classes=100),
        "ImageNet": dict(num_classes=1000),
    },
}


def get_model_parameters(model_name, dataset):
    """
    lookup model parameters by dataset

    returns a dictionary containing the keyword arguments to pass to the model constructor
    for a particular model/dataset combination.
    """
    if model_name not in DATASET_ARGUMENTS:
        raise ValueError(f"Model ({model_name}) is not in DATASET_ARGUMENTS lookup dictionary.")
    if dataset not in DATASET_ARGUMENTS[model_name]:
        raise ValueError(f"Dataset ({dataset}) is not in the DATASET_ARGUMENTS lookup for model ({model_name})")

    # get dataset specific arguments
    return DATASET_ARGUMENTS[model_name][dataset]


def get_model(model_name, build=False, dataset=None, **kwargs):
    """
    lookup model constructor from model registry by name

    if build=True, uses kwargs to build model and returns a model object
    otherwise just returns the constructor

    if build=True and dataset is not None, will look up dataset specific
    keyword arguments from the DATASET_ARGUMENTS dictionary using the
    model_name and dataset as a lookup and add those to any kwargs used
    for building the model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model ({model_name}) is not in MODEL_REGISTRY")
    model = MODEL_REGISTRY[model_name]
    if build:
        if dataset is not None:
            # get default dataset specific arguments
            dataset_specific_arguments = get_model_parameters(model_name, dataset)

            # for every dataset specific argument, if the key isn't provided in kwargs,
            # then update it using the dataset_specific_arguments
            for key, val in dataset_specific_arguments.items():
                if key not in kwargs:
                    kwargs[key] = val

        # build model with arguments
        return model(**kwargs)

    # otherwise return model constructor
    return model

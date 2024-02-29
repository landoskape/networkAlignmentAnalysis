from torch import nn
from .. import utils

# The LAYER_REGISTRY contains meta parameters for each type of layer used in alignment networks
# each layer type is associated with a few features, including:
# name (string): just for completeness, will only be used for plotting
# layer-handle (lambda method): takes as input a registered layer and returns the part of that layer
#                               to perform alignment methods on. For example, if a registered layer
#                               is layer=nn.Sequential(nn.Linear(10,10), nn.Dropout()), then the layer
#                               handle should be: lambda layer: layer[0])
# alignment_method (callable): the method used to measure alignment for a particular layer
# unfold (boolean): Determines whether the input to the layer should be flattened (e.g. if it's an image)
#                   and if the weights should be unfolded (e.g. if it's a convolutional filter)
# ignore (boolean): Determines whether the layer should be processed through alignment methods -- ignore
#                   should generally be False when a layer is used for shaping data or any other transformation
#                   not involving a weight matrix with matrix multiplication
# flag (boolean): Determines if the layer should be ignored when the "flag" switch is used. This


# Note: as of writing this, I only have nn.Linear and nn.Conv2d here, but this will start to be more
# useful and meaningful when reusing typical combinations of layers as a single registered "layer"
# that include things like dropout, pooling, nonlinearities, etc.

# NOTE NOTE: It seems like using default methods to retrieve metaparameters for compositional layers
# and just setting the name and index of the relevant layer works pretty well too

# requirements in any layer's metaparameters
REGISTRY_REQUIREMENTS = [
    "name",
    "layer_index",
    "layer_handle",
    "unfold",
    "ignore",
    "flag",
]

# lookup table for simple layer types
LAYER_REGISTRY = {
    nn.Linear: {
        "name": "linear",
        "layer_index": None,
        "layer_handle": lambda layer: layer,
        "unfold": False,
        "ignore": False,
        "flag": False,
    },
    nn.Conv2d: {
        "name": "conv2d",
        "layer_index": None,
        "layer_handle": lambda layer: layer,
        "unfold": True,
        "ignore": False,
        "flag": True,
    },
}


# a set of default metaparameters to be used for flexible layers
def default_metaprms_ignore(name):
    """convenience method for named metaparameters to be ignored"""
    metaparameters = {
        "name": name,
        "layer_index": None,
        "layer_handle": None,
        "unfold": False,
        "ignore": True,
        "flag": True,
    }
    return metaparameters


def default_metaprms_linear(index, name="linear", flag=False):
    """convenience method for named metaparameters in a linear layer packaged in a sequential"""
    metaparameters = {
        "name": name,
        "layer_index": index,
        "layer_handle": lambda layer: layer[index],
        "unfold": False,
        "ignore": False,
        "flag": flag,
    }
    return metaparameters


def default_metaprms_conv2d(index, name="conv2d", flag=True):
    """convenience method for named metaparameters in a conv2d layer packaged in a sequential"""
    metaparameters = {
        "name": name,
        "layer_index": index,
        "layer_handle": lambda layer: layer[index],
        "unfold": True,
        "ignore": False,
        "flag": flag,
    }
    return metaparameters


def check_metaparameters(metaparameters, throw=True):
    """validate whether metaparameters is a dictionary containing the required keys for an alignment network"""
    if not all([required in metaparameters for required in REGISTRY_REQUIREMENTS]):
        if throw:
            raise ValueError(
                f"metaparameters are missing required keys, it requires all of the following: {REGISTRY_REQUIREMENTS}"
            )
        return False
    return True


# Check the registry to make sure all entries are valid when importing
for layer_type, metaparameters in LAYER_REGISTRY.items():
    if not check_metaparameters(metaparameters, throw=False):
        raise ValueError(
            f"Layer type: {layer_type} from the `LAYER_REGISTRY` is missing metaparameters. "
            f"It requires all of the following: {REGISTRY_REQUIREMENTS}"
        )

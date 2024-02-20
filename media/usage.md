# Standard Usage and Documentation
Short of a documentation page, this is a simple md file that explains the main
features of the repository. For more information, please email us or raise an 
issue. 

## Key Modules
There are a few key modules that wrap pytorch objects which make scripting 
experiments smooth and consistent. Essentially, the goal is to make a script
that performs an experiment look the same even if the experiment is performed
on neural networks with extremely different structures. There's a slight
learning curve and a little less flexibility but I think that's worth it for
the clear structure that it provides. 

### Alignment Networks
The package centers around a class called ``AlignmentNetwork``, which can be
found in ``networkAlignmentAnalysis/models/base.py``. This is a child of the
pytorch ``nn.Module`` class and is used to analyze artificial neural networks
with the alignment related tools developed for this project. 

The alignment analysis are performed between the input to a layer and a weight
matrix the input activity is projected onto (whether it's a feedforward layer,
a convolutional layer, or anything else). Any neural network layer with a 
weight layer you desire to study with alignment methods should be "registered"
as a layer in the ``AlignmentNetwork``. 

``AlignmentNetworks`` have two abstract methods that are required for all
children. They are called ``initialize`` and ``get_transform_parameters``.

Here is an example for a convolutional network with two convolutional layers
and two feedforward layers built to run on the MNIST dataset (a set of 28x28, 
grayscale images with hand-written digits from 0-9). 

#### Class definition
As with any child object, the the class definition calls the parent class. 
The network is called "CNN2P2" because it has two convolutional layers and 2 
feedforward layers.
```python
class CNN2P2(AlignmentNetwork):
    """
    CNN with 2 convolutional layers, a max pooling stage, and 2 feedforward layers with dropout
    """
```

Then, the ``initialize`` method is used to define the structure of the 
network. It is called by the ``__init__`` method of the ``AlignmentNetwork`` 
base class (which passes all keyword arguments to ``initialize``).

```python
    def initialize(self, dropout=0.5):
        """architecture definition"""
        layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU())
        layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
                               nn.ReLU(), nn.MaxPool2d(kernel_size=3), nn.Flatten(start_dim=1))
        layer3 = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(256, 256), nn.ReLU())
        layer4 = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(256, 10))

        self.register_layer(layer1, **default_metaprms_conv2d(0))
        self.register_layer(layer2, **default_metaprms_conv2d(0))
        self.register_layer(layer3, **default_metaprms_linear(1))
        self.register_layer(layer4, **default_metaprms_linear(1))
```

Let's unpack what's going on here. First, four "layers" are defined in the 
usual way with the pytorch ``nn.Sequential`` class. Each layer contains a 
_single module_ with a weight parameter that is multiplied by the input to 
that layer. The module with weights is ``Conv2d`` in layers 1 and 2 and 
``Linear`` in layers 3 and 4. For clarity, I will refer to this module as the
"alignment module". 

Next, each layer is registered as a "layer" of the ``AlignmentNetwork`` using
the method ``register_layer`` inherited from the ``AlignmentNetwork`` parent
class. In addition, the second argument of ``register_layer`` defines 
metaparameters of the layer, which are explained below. Some points about 
using the ``register_layer`` method:

- It is assumed that the order of layers in the network is the same as the 
order in which they were registered. There is no flexibility here. That means
if you want the input to the network to pass through layer1 before layer2, you
have to register it first. 
- Anything can be registered as a layer as long as it is a valid ``nn.Module``
object. For example, each registered layer in this example is packaged as a
``Sequential`` object (with it's own constraints), which allows for 
flexibility with the structure of the network. However, there are constraints. 

##### Registered Layer Constraints
Any ``nn.Module`` can be added to a ``Sequential`` object representing a 
single "layer" in the ``AlignmentNetwork`` so long as it meets the following
criteria:

- If the module is added _before_ the alignment module (e.g. ``Conv2d`` or
``Linear``), then it _cannot_ change the shape of the input data. That is- 
if the output of layer N-1 has shape (B, C, W, H), then the alignment layer 
of layer N must be designed to operate on input of the same shape. For 
example, in layers 3 and 4 above, the ``Dropout`` layer does not change the 
shape of the data.
- If the next alignment layer in the network requires post-processing on the 
data, the modules that perform that post-processing _must_ be added after the
alignment layer in the _previous_ "layer". For example, convolutional layers 
require images as input, but feedforward layers require flat tensors as input. 
Therefore, the ``Flatten`` module is used after the ``Conv2d`` module of the 
last convolutional layer and before the first layer with a ``Linear`` module. 
Sorry for the similar terminology. 
- Any nonlinearities should be placed after the alignment module within a 
layer. This isn't so much a programmatic constraint as a scientific one. Email
us for more explanation. 


#### Data Transform Definition 
The code in this repository is designed to have rigid requirements for 
consistent usage while maintaining flexibility. Since each network 
architecture has different requirements for data transforms (e.g. feedforward
networks require flat input while convolutional networks require images), each
``AlignmentNetwork`` must define a method called ``get_transform_parameters``
that accepts as input a string indicating the name of the dataset and returns 
a dictionary of kwargs that will be passed to a ``DataSet`` object (see below)
to define the dataloader transform. 

Here's a simple example for the CNN2P2 network with the MNIST dataset:

```python
    def get_transform_parameters(self, dataset):
        """CNN2P2 specific transformations for each dataset"""
        params = {
            'MNIST': {
                'flatten': False,
                'resize': None, 
                'extra_transform': None, 
            }
        }
        if dataset not in params: 
            raise ValueError(f"Dataset ({dataset}) is not in params dictionary: {[k for k in params]}")
        return params[dataset]
```

The MNIST ``DataSet`` object 
[here](https://github.com/landoskape/networkAlignmentAnalysis/blob/main/networkAlignmentAnalysis/datasets.py#L135)
has the options "flatten" and "resize" as keyword arguments. For a 
convolutional network designed for MNIST, we don't need to flatten the input 
images and we don't need to resize it. (We resize in transfer learning for 
networks trained on a different dataset). The ``extra_transform`` key-value 
item is not required, but can include a callable method to be used after the
data is loaded by the dataloader. Again, see below for more explanation. 


### DataSet Objects
More to come ...


### Flags
Replicates, make multiple models so we can have error bars. List comprehension.

Checkpoints?
save_ckpts - save checkpoints



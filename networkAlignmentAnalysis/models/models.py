from torch import nn
from .base import AlignmentNetwork
from .layers import default_metaprms_conv2d, default_metaprms_ignore, default_metaprms_linear

class MLP(AlignmentNetwork):
    """
    3 hidden layer fully-connected relu network for MNIST including dropouts after input layer
    """
    def __init__(self, verbose=True):
        super().__init__()

        layer1 = nn.Sequential(nn.Linear(784, 100), nn.ReLU())
        layer2 = nn.Sequential(nn.Dropout(), nn.Linear(100, 100), nn.ReLU())
        layer3 = nn.Sequential(nn.Dropout(), nn.Linear(100, 50), nn.ReLU())
        layer4 = nn.Sequential(nn.Dropout(), nn.Linear(50, 10))

        self.register_layer(layer1, **default_metaprms_linear(0))
        self.register_layer(layer2, **default_metaprms_linear(1))
        self.register_layer(layer3, **default_metaprms_linear(1))
        self.register_layer(layer4, **default_metaprms_linear(1))

class CNN2P2(AlignmentNetwork):
    """
    CNN with 2 convolutional layers, a max pooling stage, and 2 feedforward layers with dropout
    """
    def __init__(self):
        super().__init__()

        layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU())
        layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
                               nn.ReLU(), nn.MaxPool2d(kernel_size=3), nn.Flatten(start_dim=1))
        layer3 = nn.Sequential(nn.Dropout(), nn.Linear(256, 256), nn.ReLU())
        layer4 = nn.Sequential(nn.Dropout(), nn.Linear(256, 10))

        self.register_layer(layer1, **default_metaprms_conv2d(0, each_stride=False))
        self.register_layer(layer2, **default_metaprms_conv2d(0, each_stride=False))
        self.register_layer(layer3, **default_metaprms_linear(1))
        self.register_layer(layer4, **default_metaprms_linear(1))
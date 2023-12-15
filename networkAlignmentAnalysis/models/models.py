from torch import nn
from .base import AlignmentNetwork
from .base import default_metaprms_linear, default_metaprms_conv2d, default_metaprms_ignore

class MLP(AlignmentNetwork):
    def __init__(self, verbose=True):
        super().__init__()

        layer1 = nn.Sequential(nn.Linear(784, 100), nn.ReLU())
        layer2 = nn.Sequential(nn.Dropout(), nn.Linear(100, 100), nn.ReLU())
        layer3 = nn.Sequential(nn.Dropout(), nn.Linear(100, 50), nn.ReLU())
        layer4 = nn.Sequential(nn.Dropout(), nn.Linear(50, 10))

        self.register_layer(layer1, **default_metaprms_linear(0), verbose=verbose)
        self.register_layer(layer2, **default_metaprms_linear(1), verbose=verbose)
        self.register_layer(layer3, **default_metaprms_linear(1), verbose=verbose)
        self.register_layer(layer4, **default_metaprms_linear(1), verbose=verbose)
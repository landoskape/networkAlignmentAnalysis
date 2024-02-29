from torch import nn
from torchvision.models import alexnet as torch_alexnet

from .base import AlignmentNetwork
from .layers import default_metaprms_conv2d, default_metaprms_ignore, default_metaprms_linear


class MLP(AlignmentNetwork):
    """
    3 hidden layer fully-connected relu network for MNIST including dropouts after input layer

    includes optional kwargs for changing the hidden layer widths and the input dimensionality
    """

    def initialize(self, input_dim=784, hidden_widths=[100, 100, 50], output_dim=10, dropout=0.5):
        """architecture definition"""

        # Input layer is always Linear then ReLU
        layerInput = nn.Sequential(nn.Linear(input_dim, hidden_widths[0]), nn.ReLU())

        # Hidden layers are dropout / Linear / ReLU
        layerHidden = []
        for ii in range(len(hidden_widths) - 1):
            hwin, hwout = hidden_widths[ii], hidden_widths[ii + 1]
            layerHidden.append(
                nn.Sequential(nn.Dropout(p=dropout), nn.Linear(hwin, hwout), nn.ReLU())
            )

        # Output layer is alwyays Dropout then Linear
        layerOutput = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_widths[-1], output_dim)
        )

        # Register layers in order
        self.register_layer(layerInput, **default_metaprms_linear(0))
        for layer in layerHidden:
            self.register_layer(layer, **default_metaprms_linear(1))
        self.register_layer(layerOutput, **default_metaprms_linear(1))

    def get_transform_parameters(self, dataset):
        """MLP specific transformations for each dataset"""
        params = {
            "MNIST": {
                "flatten": True,
                "resize": None,
            },
            "CIFAR10": {
                "flatten": True,
                "resize": None,
            },
            "CIFAR100": {
                "flatten": True,
                "resize": None,
            },
            "ImageNet": {
                "flatten": True,
            },
        }
        if dataset not in params:
            raise ValueError(
                f"Dataset ({dataset}) is not in params dictionary: {[k for k in params]}"
            )
        return params[dataset]


class CNN2P2(AlignmentNetwork):
    """
    CNN with 2 convolutional layers, a max pooling stage, and 2 feedforward layers with dropout

    Has flexible build method but strict 2 convolutional / max pooling / 2 feedforward architecture
    """

    def initialize(
        self,
        in_channels=1,
        output_dim=10,
        channels=[32, 64],
        kernel_size=[5, 5],
        stride=[1, 1],
        padding=[2, 2],
        num_hidden=[3136, 128],
        dropout=0.5,
        flag=True,
    ):
        """architecture definition"""

        for val, name in zip(
            (channels, kernel_size, stride, padding),
            ("channels", "kernel_size", "stride", "padding"),
        ):
            assert (
                len(val) == 2
            ), f"{name} must be 2 elements describing the parameter for each of two convolutional layers"
        assert (
            len(num_hidden) == 2
        ), "num_hidden must be 2 elements describing the number of hidden for each of two feedforward layers"

        # create layers
        layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        layer2 = nn.Sequential(
            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=kernel_size[1],
                stride=stride[1],
                padding=padding[1],
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(start_dim=1),
        )
        layer3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_hidden[0], num_hidden[1]),
            nn.ReLU(),
        )
        layer4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_hidden[1], output_dim),
        )

        self.register_layer(layer1, **default_metaprms_conv2d(0, flag=flag))
        self.register_layer(layer2, **default_metaprms_conv2d(0, flag=flag))
        self.register_layer(layer3, **default_metaprms_linear(1))
        self.register_layer(layer4, **default_metaprms_linear(1))

        # add these parameters as attributes for easy lookup later
        self.dropout = dropout

    def get_transform_parameters(self, dataset):
        """CNN2P2 specific transformations for each dataset"""
        params = {
            "MNIST": {
                "flatten": False,
                "resize": None,
            },
            "CIFAR10": {
                "flatten": False,
                "resize": None,
            },
            "CIFAR100": {
                "flatten": False,
                "resize": None,
            },
            "ImageNet": {
                "flatten": False,
            },
        }
        if dataset not in params:
            raise ValueError(
                f"Dataset ({dataset}) is not in params dictionary: {[k for k in params]}"
            )
        return params[dataset]


class AlexNet(AlignmentNetwork):
    """
    Local reimplementation of AlexNet so I can measure internal features during training without hooks

    For reference:

    AlexNet(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        (1): ReLU(inplace=True)
        (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (4): ReLU(inplace=True)
        (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace=True)
        (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): ReLU(inplace=True)
        (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
      (classifier): Sequential(
        (0): Dropout(p=0.5, inplace=False)
        (1): Linear(in_features=9216, out_features=4096, bias=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=4096, out_features=4096, bias=True)
        (5): ReLU(inplace=True)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )
    """

    def initialize(self, dropout=0.5, num_classes=1000, weights=None, flag=True):
        """architecture definition"""

        # start by loading the architecture of alexnet along with pretrained weights (if requested)
        alexnet = torch_alexnet(weights=weights, progress=True)

        # sometimes AlexNet is used for ImageNet, as it was originally conceived, sometimes it is
        # used for alternative problems. This handles updates to the output layer for the required
        # problem. (Different inputs are handled by the transforms in the dataloader).
        if num_classes == 1000:
            output_layer = alexnet.classifier[6]
        else:
            output_layer = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
            if weights is not None:
                print(
                    f"Note: you requested pretrained AlexNet weights, but are using num_output_classes={num_classes}"
                    "This means the last layer will have new (randomized) weights"
                )

        layer1 = nn.Sequential(alexnet.features[0], alexnet.features[1], alexnet.features[2])
        layer2 = nn.Sequential(alexnet.features[3], alexnet.features[4], alexnet.features[5])
        layer3 = nn.Sequential(alexnet.features[6], alexnet.features[7])
        layer4 = nn.Sequential(alexnet.features[8], alexnet.features[9])
        layer5 = nn.Sequential(
            alexnet.features[10],
            alexnet.features[11],
            alexnet.features[12],
            alexnet.avgpool,
            nn.Flatten(start_dim=1),
        )
        layer6 = nn.Sequential(alexnet.classifier[0], alexnet.classifier[1], alexnet.classifier[2])
        layer7 = nn.Sequential(alexnet.classifier[3], alexnet.classifier[4], alexnet.classifier[5])
        layer8 = nn.Sequential(output_layer)

        self.register_layer(layer1, **default_metaprms_conv2d(0, flag=flag))
        self.register_layer(layer2, **default_metaprms_conv2d(0, flag=flag))
        self.register_layer(layer3, **default_metaprms_conv2d(0, flag=flag))
        self.register_layer(layer4, **default_metaprms_conv2d(0, flag=flag))
        self.register_layer(layer5, **default_metaprms_conv2d(0, flag=flag))
        self.register_layer(layer6, **default_metaprms_linear(1))
        self.register_layer(layer7, **default_metaprms_linear(1))
        self.register_layer(layer8, **default_metaprms_linear(0))

        # add these parameters as attributes for easy lookup later
        self.dropout = dropout

        # set dropout with general method so we can easily use each alexnet.features/classifier/etc
        self.set_dropout(dropout)

    def get_transform_parameters(self, dataset):
        """Alexnet specific transformations for each dataset"""

        def gray_to_rgb(batch):
            batch[0] = batch[0].expand(-1, 3, -1, -1)
            return batch

        params = {
            "MNIST": {
                "flatten": False,
                "resize": (256, 256),
                # 'out_channel': 3, # will replace extra-transform with this when torchvision is ready
                "extra_transform": gray_to_rgb,
            },
            "CIFAR10": {
                "flatten": False,
                "resize": (256, 256),
            },
            "CIFAR100": {
                "flatten": False,
                "resize": (256, 256),
            },
            "ImageNet": {
                "center_crop": 224,
                "flatten": False,
                "resize": (256, 256),
            },
        }

        if dataset not in params:
            raise ValueError(
                f"Dataset ({dataset}) is not in params dictionary: {[k for k in params]}"
            )
        return params[dataset]

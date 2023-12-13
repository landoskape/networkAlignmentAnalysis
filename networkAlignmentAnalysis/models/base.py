from torch import nn

LAYER_REGISTRY = {
    nn.Linear: ('linear', lambda layer: layer.weight.data),
    nn.Conv2d: ('conv2d', lambda layer: layer.weight.data),
}

class BaseNetwork(nn.Module):
    """
    This is the base class for a neural network used for alignment-related 
    experiments. The goal is for networks with specific functions to inherit
    methods and properties from the **BaseNetwork** that allow seamless 
    integration with the general tools required for the project
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList() # a list of all modules in the forward pass
        self.layer_types = [] # list of strings describing the type of the layer
        self.layer_weight_handle = [] # list of handles to retrieve the relevant weights of the layer
        self.hidden = [] # list of tensors containing hidden activations

    def register_layer(self, layer, weight_handle=None, verbose=True):
        """
        register_layer adds a **layer** to the network's module list
        the **type** determines what kind of alignment related processing is done on the layer
        """
        layer_type, default_weight_handle = LAYER_REGISTRY.get(type(layer), (None, None))
        
        if not isinstance(layer, nn.Module):
            raise TypeError(f"provided layer is of type: {type(layer)}, but only nn.Module objects are permitted!")
        
        layer_weight_handle = weight_handle if weight_handle is not None else default_weight_handle
        
        # add layer to network
        self.layers.append(layer)
        self.layer_types.append(layer_type)
        self.layer_weight_handle.append(layer_weight_handle)

        if verbose:
            if layer_type is not None:
                print(f"Added a {layer_type} layer to the network with the following properties:\n {layer}")
            else:
                print(f"Added an unknown layer type to the network, it will be used in the forward pass but not processed. Layer props:\n {layer}")

    def forward(self, x, store_hidden=False):
        self.hidden = [] # always reset so as to not keep a previous forward pass accidentally
        for layer in self.layers:
            x = layer(x) # pass through next layer
            if store_hidden: self.hidden.append(x)
        return x
    

    

    
import numpy as np
import torch
from torch import nn
from .layers import LAYER_REGISTRY, REGISTRY_REQUIREMENTS, check_metaparameters
from ..utils import check_iterable

class AlignmentNetwork(nn.Module):
    """
    This is the base class for a neural network used for alignment-related experiments. 

    The point of all the wrangling of standard torch workflows in this class is to make 
    it easy to perform all the alignment-related computations for networks with different
    architectures without having to rewrite similar code over and over again. In this way,
    the user only needs to add a layer type to the **LAYER_REGISTRY** in this file and
    then alignment methods can be automatically applied. 

    The forward method of **AlignmentNetwork** passes the input (*x*) through each registered
    layer of the network in order of it's registration. If hidden activations are requested, then
    the output of each registered layer is saved. The alignment methods are applied to the 
    hidden activation at the output of layer L-1 and the weights of layer L. 

    Note: some shape wrangling (like that which happens between a convolutional layer and a
    linear layer are often treated as a nn.Module layer), but these don't require alignment-
    related processing. To use these, set 'ignore' of the metaparameters to True. Alternatively,
    you can append them to the last component of a layer.

    Note: the choice of where to put layers matters. I usually nest a dropout in a sequential
    layer in front of whatever layer is relevant for alignment, because then the previous 
    hidden outputs will be measured before using dropout. Of course, this may not be desired,
    it's just about your scientific question. 

    A layer in the layer_registry should have the following properties:
    1. Be a child of the nn.Module class with a forward method
    2. Have at most one "relevant" processing stage with weights for measuring alignment
    """
    def __init__(self):
        super().__init__() # register it as a nn.Module
        self.layers = nn.ModuleList() # a list of all modules in the forward pass
        self.metaparameters = [] # list of dictionaries containing metaparameters for each layer
        self.hidden = [] # list of tensors containing hidden activations

    def register_layer(self, layer, **kwargs):
        """
        register_layer adds a **layer** to the network's module list and it's associated metaparameters
        for determining what kind of aligment-related processing is done on the layer

        by default, the layer is used as a key to lookup the metaparameters from the **LAYER_REGISTRY**. 
        kwargs can update keys in the metaparameters. If the layer class is not registered, then all 
        metaparameters must be provided as kwargs.
         
        Required kwargs are: name, layer_handle, alignment_method, ignore, ...
        """
        if not isinstance(layer, nn.Module):
            raise TypeError(f"provided layer is of type: {type(layer)}, but only nn.Module objects are permitted!")
        
        metaparameters = LAYER_REGISTRY.get(type(layer), {})
        for metaprms in REGISTRY_REQUIREMENTS:
            # for each possible entry in layer metaparameters, check if it's provided, not none, then update it
            if metaprms in kwargs and kwargs[metaprms] is not None:
                metaparameters[metaprms]=kwargs[metaprms]
        
        # check whether metaparameters contain the correct keys
        check_metaparameters(metaparameters, throw=True)
        
        # add layer to network
        self.layers.append(layer)
        self.metaparameters.append(metaparameters)


    def forward(self, x, store_hidden=False):
        """standard forward pass of all layers with option of storing hidden activations (and output)"""
        self.hidden = [] # always reset so as to not keep a previous forward pass accidentally
        for layer, metaprms in zip(self.layers, self.metaparameters):
            x = layer(x) # pass through next layer
            if store_hidden and not metaprms['ignore']: 
                self.hidden.append(x)
        return x
    
    @torch.no_grad()
    def get_activations(self, x=None, precomputed=False):
        """convenience method for getting list of intermediate activations throughout the network"""
        if not precomputed:
            if x is None:
                raise ValueError("x needs to be provided if precomputed is False")
            else: 
                _ = self.forward(x, store_hidden=True)
        return self.hidden
    
    @torch.no_grad()
    def get_alignment_layers(self):
        """convenience method for retrieving registered layers for alignment measurements throughout the network"""
        layers = []
        for layer, metaprms in zip(self.layers, self.metaparameters):
            if not metaprms['ignore']:
                layers.append(metaprms['layer_handle'](layer))
        return layers
    
    @torch.no_grad()
    def get_alignment_metaparameters(self):
        """convenience method for retrieving registered layers for alignment measurements throughout the network"""
        metaparameters = []
        for metaprms in self.metaparameters:
            if not metaprms['ignore']:
                metaparameters.append(metaprms)
        return metaparameters
    
    @torch.no_grad()
    def get_alignment_weights(self):
        """convenience method for retrieving registered weights for alignment measurements throughout the network"""
        return [layer.weight.data for layer in self.get_alignment_layers()]
    
    @torch.no_grad()
    def compare_weights(self, weights):
        current_weights = self.get_alignment_weights()
        delta_weights = []
        for iw, cw in zip(weights, current_weights):
            delta_weights.append(torch.norm(cw.flatten(1)-iw.flatten(1), dim=1))
        return delta_weights
    
    @torch.no_grad()
    def measure_alignment(self, x, precomputed=False, method='alignment'):
        # Pre-layer activations start with input (x and ignore output)
        activations = [x, *self.get_activations(x=x, precomputed=precomputed)[:-1]]
        alignment = []
        for activation, layer, metaprms in zip(activations, self.get_alignment_layers(), self.get_alignment_metaparameters()):
            alignment.append(metaprms['alignment_method'](activation, layer, method=method))
        return alignment

    @torch.no_grad()
    def forward_targeted_dropout(self, x, idxs=None, layers=None):
        """
        perform forward pass with targeted dropout of hidden channels

        **idxs** and **layers** are matched length tuples describing the layer to dropout
        in and the idxs in that layer to dropout. The dropout happens after the layer (so
        layer=(0) will correspond to the output of the first layer.

        returns the output accounting for targeted dropout and also the full list of hidden
        activations after targeted dropout (can use forward with store_hidden=True) for 
        hidden activations without targeted dropout and use self.eval() for no dropout at all
        """
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers need to be iterables with the same length"
        assert len(idxs)==len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers)==len(np.unique(layers)), "layers must not have any repeated elements"
        assert all([layer>=0 and layer<len(self.layers)-1 for layer in layers]), "dropout only works on first N-1 layers"
        
        activations = []
        for idx_layer, (layer, metaprms) in enumerate(zip(self.layers, self.metaparameters)):
            x = layer(x) # pass through next layer
            if idx_layer in layers:
                dropout_idx = idxs[{val:idx for idx, val in enumerate(layers)}[idx_layer]]
                fraction_dropout = len(dropout_idx) / x.shape[1]
                x[:, dropout_idx] = 0
                x = x * (1 - fraction_dropout)
            if not metaprms['ignore']:
                activations.append(x) 
            
        return x, activations
    
    @torch.no_grad()
    def measure_eigenfeatures(self, dataloader, DEVICE=None):
        if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Measure Activations (without dropout) for all images
        training_mode = self.training # save for resetting at the end

        # turn off dropout and any training related features
        self.eval()

        # store input and measure activations for every element in dataloader
        allinputs = []
        activations = []
        for images, label in dataloader:    
            allinputs.append(images)
            input = input.to(DEVICE)
            label = label.to(DEVICE)
            activations.append(self.get_activations(x=input, precomputed=False))

        # return network to original training mode
        if training_mode: 
            self.train()
        else:
            self.eval()

        # create large list of tensors containing input to each layer
        num_layers = len(self.layers)
        inputs_to_layers = []
        inputs_to_layers.append(torch.cat(allinputs,dim=0).detach().cpu())
        for layer in range(num_layers-1):
            inputs_to_layers.append(torch.cat([cact[layer] for cact in activations],dim=0).detach().cpu())

        # Measure eigenfeatures of input to each layer
        eigenvalues = []
        eigenvectors = []
        for itl in inputs_to_layers:
            # covariance matrix is positive semidefinite, but numerical errors can produce negative eigenvalues
            ccov = torch.cov(itl.T)
            crank = torch.linalg.matrix_rank(ccov)
            w, v = torch.linalg.eigh(ccov)
            w_idx = torch.argsort(-w)
            w = w[w_idx]
            v = v[:,w_idx]

            # Automatically set eigenvalues to 0 when they are numerical errors!
            w[crank:] = 0
            eigenvalues.append(w)
            eigenvectors.append(v)

        # Measure dot product of weights on eigenvectors for each layer
        beta = []
        netweights = self.get_alignment_weights()
        for evc,nw in zip(eigenvectors,netweights):
            nw = nw / torch.norm(nw,dim=1,keepdim=True)
            beta.append(torch.abs(nw.cpu() @ evc))
            
        return beta, eigenvalues, eigenvectors
    

# def ExperimentalNetwork(AlignmentNetwork):
#     """maintain some experimental methods here"""
#     def doOjaUpdate(self, x, alpha):
#         # Rule: dW = alpha * (xy - wy**2) 
#         B = x.shape[0]
#         activations = self.getActivations(x)
#         # Layer 1:
#         H,D = (activations[0].shape[1], x.shape[1])
#         dfc1 = alpha * (activations[0].T @ x - torch.sum(self.fc1.weight.data.clone().detach().reshape(H,D,1) * (activations[0]*2).T.reshape(H,B,1).permute(0,2,1),dim=2))
#         self.fc1.weight.data = self.fc1.weight.data + dfc1
#         self.fc1.weight.data = self.fc1.weight.data / torch.norm(self.fc1.weight.data,dim=1,keepdim=True)
#         #print(f"fc1: Weight.shape:{self.fc1.weight.data.shape}, update.shape:{dfc1.shape}")
#         # Layer 2:
#         H,D = (activations[1].shape[1], activations[0].shape[1])
#         dfc2 = alpha * (activations[1].T @ activations[0] - torch.sum(self.fc2.weight.data.clone().detach().reshape(H,D,1) * (activations[1]*2).T.reshape(H,B,1).permute(0,2,1),dim=2))
#         self.fc2.weight.data = self.fc2.weight.data + dfc2
#         self.fc2.weight.data = self.fc2.weight.data / torch.norm(self.fc2.weight.data,dim=1,keepdim=True)
#         #print(f"fc2: Weight.shape:{self.fc2.weight.data.shape}, update.shape:{dfc2.shape}")
#         # Layer 3:
#         H,D = (activations[2].shape[1], activations[1].shape[1])
#         dfc3 = alpha * (activations[2].T @ activations[1] - torch.sum(self.fc3.weight.data.clone().detach().reshape(H,D,1) * (activations[2]*2).T.reshape(H,B,1).permute(0,2,1),dim=2))
#         self.fc3.weight.data = self.fc3.weight.data + dfc3
#         self.fc3.weight.data = self.fc3.weight.data / torch.norm(self.fc3.weight.data,dim=1,keepdim=True)
#         #print(f"fc3: Weight.shape:{self.fc3.weight.data.shape}, update.shape:{dfc3.shape}")
#         # Layer 4:
#         H,D = (activations[3].shape[1], activations[2].shape[1])
#         dfc4 = alpha * (activations[3].T @ activations[2] - torch.sum(self.fc4.weight.data.clone().detach().reshape(H,D,1) * (activations[3]*2).T.reshape(H,B,1).permute(0,2,1),dim=2))
#         self.fc4.weight.data = self.fc4.weight.data + dfc4
#         self.fc4.weight.data = self.fc4.weight.data / torch.norm(self.fc4.weight.data,dim=1,keepdim=True)
#         #print(f"fc4: Weight.shape:{self.fc4.weight.data.shape}, update.shape:{dfc4.shape}")

#     def manualShape(self,evals,evecs,DEVICE,evalTransform=None):
#         if evalTransform is None:
#             evalTransform = lambda x:x
            
#         sbetas = [] # produce signed betas
#         netweights = self.getNetworkWeights()
#         for evc,nw in zip(evecs,netweights):
#             nw = nw / torch.norm(nw,dim=1,keepdim=True)
#             sbetas.append(nw.cpu() @ evc)
        
#         shapedWeights = [[] for _ in range(self.numLayers)]
#         for layer in range(self.numLayers):
#             assert np.all(evals[layer]>=0), "Found negative eigenvalues..."
#             cFractionVariance = evals[layer]/np.sum(evals[layer]) # compute fraction of variance explained by each eigenvector
#             cKeepFraction = evalTransform(cFractionVariance).astype(cFractionVariance.dtype) # make sure the datatype doesn't change, otherwise pytorch einsum will be unhappy
#             assert np.all(cKeepFraction>=0), "Found negative transformed keep fractions. This means the transform function has an improper form." 
#             assert np.all(cKeepFraction<=1), "Found keep fractions greater than 1. This is bad practice, design the evalTransform function to have a domain and range within [0,1]"
#             weightNorms = torch.norm(netweights[layer],dim=1,keepdim=True) # measure norm of weights (this will be invariant to the change)
#             evecComposition = torch.einsum('oi,xi->oxi',sbetas[layer],torch.tensor(evecs[layer])) # create tensor composed of each eigenvector scaled to it's contribution in each weight vector
#             newComposition = torch.einsum('oxi,i->ox',evecComposition,torch.tensor(cKeepFraction)).to(DEVICE) # scale eigenvectors based on their keep fraction (by default scale them by their variance)
#             shapedWeights[layer] = newComposition / torch.norm(newComposition,dim=1,keepdim=True) * weightNorms
        
#         # Assign new weights to network
#         self.fc1.weight.data = shapedWeights[0]
#         self.fc2.weight.data = shapedWeights[1]
#         self.fc3.weight.data = shapedWeights[2]
#         self.fc4.weight.data = shapedWeights[3]


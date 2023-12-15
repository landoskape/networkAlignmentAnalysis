# import numpy as np
# import scipy as sp
import torch
from torch import nn
from .. import utils
from .layers import LAYER_REGISTRY, REGISTRY_REQUIREMENTS, check_metaparameters

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

#     def targetedDropout(net,x,idx=None,layer=None,returnFull=False):
#         assert layer>=0 and layer<=2, "dropout only works on first three layers"
#         h1 = net.actFunc(net.fc1(x))
#         if layer==0: 
#             fracDropout = len(idx)/h1.shape[1]
#             h1[:,idx]=0
#             h1 = h1 * (1 - fracDropout)
#         h2 = net.actFunc(net.fc2(h1))
#         if layer==1: 
#             fracDropout = len(idx)/h2.shape[1]
#             h2[:,idx]=0
#             h2 = h2 * (1 - fracDropout)            
#         h3 = net.actFunc(net.fc3(h2))
#         if layer==2: 
#             fracDropout = len(idx)/h3.shape[1]
#             h3[:,idx]=0
#             h3 = h3 * (1 - fracDropout)
#         out = net.fc4(h3)
#         if returnFull: return h1,h2,h3,out
#         else: return out
    
#     def mlTargetedDropout(net,x,idx,layer,returnFull=False):
#         assert type(idx) is tuple and type(layer) is tuple, "idx and layer need to be tuples"
#         assert len(idx)==len(layer), "idx and layer need to have the same length"
#         npLayer = np.array(layer)
#         assert len(npLayer)==len(np.unique(npLayer)), "layer must not have any repeated elements"
#         # Do forward pass with targeted dropout
#         h1 = net.actFunc(net.fc1(x))
#         if np.any(npLayer==0):
#             cIndex = idx[npLayer==0]
#             fracDropout = len(cIndex)/h1.shape[1]
#             h1[:,cIndex]=0
#             h1 = h1 * (1 - fracDropout)
#         h2 = net.actFunc(net.fc2(h1))
#         if np.any(npLayer==1):
#             cIndex = idx[npLayer==1]
#             fracDropout = len(cIndex)/h2.shape[1]
#             h2[:,cIndex]=0
#             h2 = h2 * (1 - fracDropout)            
#         h3 = net.actFunc(net.fc3(h2))
#         if np.any(npLayer==2):
#             cIndex = idx[npLayer==2]
#             fracDropout = len(cIndex)/h3.shape[1]
#             h3[:,cIndex]=0
#             h3 = h3 * (1 - fracDropout)
#         out = net.fc4(h3)
#         if returnFull: return h1,h2,h3,out
#         else: return out
    
#     @staticmethod
#     def measureEigenFeatures(net, dataloader, DEVICE=None):
#         # Handle DEVICE if not provided
#         if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Measure Activations (without dropout) for all images
#         storeDropout = net.getDropout()
#         net.setDropout(0) # no dropout for measuring eigenfeatures
#         allimages = []
#         activations = []
#         for images, label in dataloader:    
#             allimages.append(images)
#             images = images.to(DEVICE)
#             label = label.to(DEVICE)
#             activations.append(net.getActivations(images))
#         net.setDropout(storeDropout)

#         # Consolidate variable structure
#         NL = net.numLayers
#         allinputs = []
#         allinputs.append(torch.cat(allimages,dim=0).detach().cpu())
#         for layer in range(NL-1):
#             allinputs.append(torch.cat([cact[layer] for cact in activations],dim=0).detach().cpu())

#         # Measure eigenfeatures for each layer
#         eigenvalues = []
#         eigenvectors = []
#         for ai in allinputs:
#             # Covariance matrix is positive semidefinite, but numerical errors can produce negative eigenvalues
#             ccov = torch.cov(ai.T)
#             crank = torch.linalg.matrix_rank(ccov)
#             w,v = sp.linalg.eigh(ccov)
#             widx = np.argsort(w)[::-1]
#             w = w[widx]
#             v = v[:,widx]
#             # Automatically set eigenvalues to 0 when they are numerical errors!
#             w[crank:]=0
#             eigenvalues.append(w)
#             eigenvectors.append(v)

#         # Measure dot product of weights on eigenvectors for each layer
#         beta = []
#         netweights = net.getNetworkWeights()
#         for evc,nw in zip(eigenvectors,netweights):
#             nw = nw / torch.norm(nw,dim=1,keepdim=True)
#             beta.append(torch.abs(nw.cpu() @ evc))
            
#         return beta, eigenvalues, eigenvectors
    
from math import prod
from tqdm import tqdm
from abc import ABC, abstractmethod 
import torch
from torch import nn

from warnings import warn

from .layers import LAYER_REGISTRY, REGISTRY_REQUIREMENTS, check_metaparameters
from ..utils import check_iterable, get_maximum_strides, batch_cov, named_transpose, weighted_average, get_device

class AlignmentNetwork(nn.Module, ABC):
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
    def __init__(self, **kwargs):
        super().__init__() # register it as a nn.Module
        self.layers = nn.ModuleList() # a list of all modules in the forward pass
        self.metaparameters = [] # list of dictionaries containing metaparameters for each layer
        self.hidden = [] # list of tensors containing hidden activations
        self.ignore_flag = kwargs.pop('ignore_flag', False) # setting for whether to ignore flagged layers
        self.initialize(**kwargs) # initialize the architecture using child class method

    @abstractmethod
    def initialize(self, **kwargs):
        """
        initialize is defined by child objects of the AlignmentNetwork class and is where
        the network architecture is established

        generally, initialize methods create a set of layers that represent the network
        and use the `register_layer` method to add them to the network.

        depending on the child network, various kwargs are required to initialize the network
        these are passed into the class constructor and relayed to initialize. See each class
        definition's initialize method for the specific kwargs relevant to each network type
        """
        pass

    @abstractmethod
    def get_transform_parameters(self, dataset):
        """
        this method is required throughout the repository to load specific transform parameters
        for each dataset for each network. In each child class of AlignmentNetwork, this method
        should define a dictionary where the possible datasets are the keys (as strings) and the
        value for each dataset-key is another dictionary passed into the DataSet constructor to
        define the transform parameters
        """
        pass

    def register_layer(self, layer, **kwargs):
        """
        register_layer adds a **layer** to the network's module list and it's associated metaparameters
        for determining what kind of aligment-related processing is done on the layer

        by default, the layer is used as a key to lookup the metaparameters from the **LAYER_REGISTRY**. 
        kwargs can update keys in the metaparameters. If the layer class is not registered, then all 
        metaparameters must be provided as kwargs.
         
        Required kwargs are: name, layer_handle, alignment_method, unfold, ignore, ...
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

    def num_layers(self, all=False):
        """
        convenience method for getting the number of layers in network
        if all=False (default), will get the number of alignment layers
        if all=True, will get total number of registered layers in network
        """
        if all: 
            return len(self.layers)
        return len(self.get_alignment_layers())
    
    def _include_layer(self, metaprms):
        """
        internal method for checking whether to include a registered layer

        ignore conditions:
        if metaprms['ignore']==True, will always ignore the layer
        if metaprms['flag']==True and self.ignore_flag==True, will ignore
        """
        return not metaprms['ignore'] and not(metaprms['flag'] and self.ignore_flag)
    
    def set_ignore_flag(self, ignore_flag):
        """method for setting the ignore_flag switch"""
        assert isinstance(ignore_flag, bool), "ignore_flag setting must be a bool"
        self.ignore_flag = ignore_flag

    def forward(self, x, store_hidden=False):
        """standard forward pass of all layers with option of storing hidden activations (and output)"""
        self.hidden = [] # always reset so as to not keep a previous forward pass accidentally
        for layer in self.layers:
            x = layer(x) # pass through next layer
            if store_hidden:
                self.hidden.append(x)
        return x
    
    def get_dropout(self): 
        """
        Return list of dropout probability for any dropout layers in network
        """
        p = []
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                p.append(module.p)
        return p
    
    def set_dropout(self, p):
        """
        Set dropout of all layers in a network
        Note that this will overwrite whatever was previously used
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = p

        if hasattr(self, 'dropout'):
            setattr(self, 'dropout', p)
    
    def set_dropout_by_layer(self, p):
        """
        Set dropout of each layer in a network independently

        p must be an iterable indicating the probability of dropout for each layer
        """
        # get dropout layers (in order!)
        dropout_layers = []
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(module)
        
        assert len(dropout_layers) == len(p), "p must contain the same number of elements as the number of dropout layers in the network"

        # assign each p to the dropout layer
        for layer, drop_prob in zip(dropout_layers, p):
            layer.p = drop_prob

        # set dropout attribute if it exists
        if hasattr(self, 'dropout'):
            setattr(self, 'dropout', p)
        
    @torch.no_grad()
    def get_layer_outputs(self, x=None, precomputed=False):
        """method for getting list of layer outputs throughout the network"""
        if not precomputed:
            if x is None:
                raise ValueError("x needs to be provided if precomputed is False")
            else: 
                # do a forward pass and store hidden activations if not precomputed
                _ = self.forward(x, store_hidden=True)
        
        # filter activations by outputs of alignment layers 
        layer_outputs = []
        for hidden, metaprms in zip(self.hidden, self.metaparameters):
            if self._include_layer(metaprms):
                layer_outputs.append(hidden)

        # return outputs
        return layer_outputs
    
    @torch.no_grad()
    def get_layer_inputs(self, x, precomputed=False):
        """method for getting list of layer inputs throughout the network"""
        if not precomputed:
            # do a forward pass and store hidden activations if not precomputed
            _ = self.forward(x, store_hidden=True)
        
        # filter activations by inputs to alignment layers 
        layer_inputs = []
        possible_inputs = [x, *self.hidden[:-1]]
        for inputs, metaprms in zip(possible_inputs, self.metaparameters):
            if self._include_layer(metaprms):
                layer_inputs.append(inputs)

        # return outputs
        return layer_inputs
    
    @torch.no_grad()
    def get_alignment_layers(self):
        """convenience method for retrieving registered layers for alignment measurements throughout the network"""
        layers = []
        for layer, metaprms in zip(self.layers, self.metaparameters):
            if self._include_layer(metaprms):
                layers.append(metaprms['layer_handle'](layer))
        return layers
    
    @torch.no_grad()
    def get_alignment_metaparameters(self):
        """convenience method for retrieving registered layers for alignment measurements throughout the network"""
        metaparameters = []
        for metaprms in self.metaparameters:
            if self._include_layer(metaprms):
                metaparameters.append(metaprms)
        return metaparameters
    
    @torch.no_grad()
    def get_alignment_layer_indices(self):
        """convenience method for retrieving the absolute indices of alignment layers throughout the network"""
        idx_layers = []
        for idx, metaprms in enumerate(self.metaparameters):
            if self._include_layer(metaprms):
                idx_layers.append(idx)
        return idx_layers
    
    @torch.no_grad()
    def get_alignment_weights(self, with_unfold=False, input_to_layers=None):
        """
        convenience method for retrieving registered weights for alignment measurements throughout the network
        
        optionally unfold convolutional weights if requested (by **with_unfold** set to True), this requires 
        the **input_to_layers** to be provided to determine how many strides to unfold for
        """
        if with_unfold and input_to_layers is None:
            raise ValueError("If unfolding is requested, input_to_layers must be provided.")
        
        # get weights
        weights = [layer.weight.data for layer in self.get_alignment_layers()]

        # unfold weights if requested
        if with_unfold: 
            if len(input_to_layers) != len(weights):
                raise ValueError(f"There are {len(weights)} registered weights but only {len(input_to_layers)} were provided.")

            # measure number of strides per layer
            layers = self.get_alignment_layers()
            num_strides = [prod(get_maximum_strides(input.shape[2], input.shape[3], layer))
                           for input, layer in zip(input_to_layers, layers)]
            
            # then unfold and repeat weights for each stride
            weights = [weight.view(weight.size(0), -1).repeat(1, strides)
                       for weight, strides in zip(weights, num_strides)]

        return weights
    
    @torch.no_grad()
    def compare_weights(self, weights):
        current_weights = self.get_alignment_weights()
        delta_weights = []
        for iw, cw in zip(weights, current_weights):
            delta_weights.append(torch.norm(cw.flatten(1)-iw.flatten(1), dim=1))
        return delta_weights
    
    @torch.no_grad()
    def measure_alignment(self, x, precomputed=False, method='alignment'):
        # Pre-layer activations start with input (x) and ignore output
        layer_inputs = self.get_layer_inputs(x, precomputed=precomputed)
        alignment = []
        for input, layer, metaprms in zip(layer_inputs, self.get_alignment_layers(), self.get_alignment_metaparameters()):
            alignment.append(metaprms['alignment_method'](input, layer, method=method))
        return alignment
    
    @torch.no_grad()
    def measure_correlation(self, x, precomputed=False, method='corr', reduced=True):
        correlation = []
        zipped = zip(self.get_layer_outputs(x=x, precomputed=precomputed), self.get_alignment_metaparameters())
        for activation, metaprms in zipped:
            ccorr = metaprms['correlation_method'](activation, method=method)
            if reduced:
                ccorr = torch.mean(torch.abs(ccorr))
            correlation.append(ccorr)
        return correlation

    @torch.no_grad()
    def forward_targeted_dropout(self, x, idxs=None, layers=None):
        """
        perform forward pass with targeted dropout of hidden channels

        **idxs** and **layers** are matched length tuples describing the layer to dropout
        in and the idxs in that layer to dropout. The dropout happens in the activations 
        of the layer (so layer=(0) corresponds to the output of the first layer).

        returns the output accounting for targeted dropout and also the full list of hidden
        activations after targeted dropout
        """
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers need to be iterables with the same length"
        assert len(idxs)==len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers)==len(set(layers)), "layers must not have any repeated elements"
        assert all([layer>=0 and layer<len(self.layers)-1 for layer in layers]), "dropout only works on first N-1 layers"
        
        activations = []
        for idx_layer, (layer, metaprms) in enumerate(zip(self.layers, self.metaparameters)):
            x = layer(x) # pass through next layer
            if idx_layer in layers:
                dropout_idx = idxs[{val:idx for idx, val in enumerate(layers)}[idx_layer]]
                fraction_dropout = len(dropout_idx) / x.shape[1]
                x[:, dropout_idx] = 0
                x = x * (1 - fraction_dropout)
            if self._include_layer(metaprms):
                activations.append(x) 
            
        return x, activations
    
    @torch.no_grad()
    def forward_eigenvector_dropout(self, x, eigenvectors, idxs, layers):
        """
        perform forward pass with targeted dropout of activity on eigenvectors 

        **eigenvectors**, **idxs** and **layers** are matched length tuples describing the
        layer to dropout in, the eigenvectors of input to that layer, and the idxs of which
        eigenvectors in that layer to dropout. The dropout happens after the layer (so 
        layer=(0) will correspond to the output of the first layer.

        returns the output accounting for targeted dropout and also the full list of hidden
        activations after targeted dropout
        """
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers need to be iterables with the same length"
        assert len(idxs)==len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers)==len(set(layers)), "layers must not have any repeated elements"
        assert all([layer>=0 and layer<len(self.layers)-1 for layer in layers]), "dropout only works on first N-1 layers"
        
        activations = []
        for idx_layer, (layer, metaprms) in enumerate(zip(self.layers, self.metaparameters)):
            x = layer(x) # pass through next layer
            if idx_layer in layers:
                dropout_idx = idxs[{val:idx for idx, val in enumerate(layers)}[idx_layer]]
                fraction_dropout = len(dropout_idx) / x.shape[1]
                x[:, dropout_idx] = 0
                x = x * (1 - fraction_dropout)
            if self._include_layer(metaprms):
                activations.append(x) 
            
        return x, activations
    
    @torch.no_grad()
    def measure_eigenfeatures(self, dataloader, DEVICE=None, with_updates=True, full_conv=False):
        """
        measure the eigenvalues and eigenvectors of the input to each layer
        and also measure how much each weight array uses each eigenvector

        computing eigenfeatures is intensive for big matrices so it's not a 
        good idea to this on unfolded data in convolutional layers. It may be 
        a good idea to do it sometimes -- I think sklearn's IncrementalPCA 
        algorithm is best for this. But it still takes a while so shouldn't be
        done frequently, only after training for important networks. 

        **full_conv** is used to determine whether to unfold convolutional layers
        -- if full_conv=True, will unfold and measure eigenfeatures that way
        -- if full_conv=False, will measure for each stride (and take the average
        across strides weighted by the variance of the input data)
        """
        # Set device automatically if not provided
        if DEVICE is None: 
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Measure Activations (without dropout) for all images
        training_mode = self.training # save for resetting at the end

        # turn off dropout and any training related features
        self.eval()

        # store input and measure activations for every element in dataloader
        allinputs = []
        dataloop = tqdm(dataloader) if with_updates else dataloader
        for input, label in dataloop:
            input = input.to(DEVICE)
            allinputs.append(self.get_layer_inputs(input, precomputed=False))

        # return network to original training mode
        if training_mode: 
            self.train()
        else:
            self.eval()

        # create large list of tensors containing input to each layer
        inputs_to_layers = [torch.cat([input[layer] for input in allinputs], dim=0).detach().cpu()
                            for layer in range(self.num_layers())]
        
        # retrieve weights and flatten inputs if required
        weights = []
        metaprms = self.get_alignment_metaparameters()
        zipped = zip(inputs_to_layers, self.get_alignment_layers(), metaprms)
        for ii, (input, layer, metaprm) in enumerate(zipped):
            weight = layer.weight.data # get alignment layer weight data
            if metaprm['unfold']:
                # get unfolded input activity
                layer_prms = dict(stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
                unfolded_input = torch.nn.functional.unfold(input, layer.kernel_size, **layer_prms)
                if full_conv:
                    # flatten unfolded input
                    unfolded_input = unfolded_input.transpose(1, 2).reshape(input.size(0), -1)
                    inputs_to_layers[ii] = unfolded_input

                    # unfold weights and add them to weights list
                    num_strides = prod(get_maximum_strides(input.shape[2], input.shape[3], layer))
                    unfolded_weights = weight.view(weight.size(0), -1).repeat(1, num_strides)
                    weights.append(unfolded_weights)

                else:
                    # keep unfolded input activity in (dim x stride) format
                    inputs_to_layers[ii] = unfolded_input

                    # flatten weights (without repeating!)
                    flat_weights = weight.view(weight.size(0), -1)
                    weights.append(flat_weights)

            else:
                # if not unfolding weights, just add them directly
                weights.append(weight)

        # Measure eigenfeatures of input to each layer
        beta = []
        eigenvalues = []
        eigenvectors = []
        zipped = zip(inputs_to_layers, weights, metaprms)
        for input, weight, metaprm in zipped:
            if not full_conv and metaprm['unfold']:
                # measure variance across samples (each batch element of the dataset)
                # average variance across dimensions of weight across each stride
                bvar = torch.mean(torch.var(input, dim=0), dim=0) 

                # measuring across each stride independently (for conv layers)
                bcov = batch_cov(input.permute((2, 1, 0)))
                brank = [torch.linalg.matrix_rank(bc) for bc in bcov]
                
                # eigh will fail if condition number is too high (which can happend
                # in these strided input activity). If that's the case, we set the 
                # covariance to the identity so eigh will work, and set the variance
                # to 0 in that stride, so the weighted_average will ignore that stride.
                for ii, bc in enumerate(bcov):
                    if torch.isinf(torch.linalg.cond(bc)):
                        bvar[ii] = 0 # set variance to 0 to ignore this dimension
                        bcov[ii] = torch.eye(bcov.size(1))
                
                # measure eigenvalues and eigenvectors
                w, v = named_transpose([torch.linalg.eigh(bc) for bc in bcov])
                w_idx = [torch.argsort(-ww) for ww in w]
                w = [ww[wi] for ww, wi in zip(w, w_idx)]
                v = [vv[:, wi] for vv, wi in zip(v, w_idx)]
                for ii, br in enumerate(brank):
                    w[ii][br:] = 0
                
                # stack eigenvectors and eigenvalues into tensor
                w = torch.stack(w)
                v = torch.stack(v)

                # Measure abs value of dot product of weights on eigenvectors for each layer
                num_strides = v.size(0)
                weight = weight / torch.norm(weight, dim=1, keepdim=True)
                b = torch.bmm(weight.cpu().unsqueeze(0).expand(num_strides, -1, -1), v)

                # Contract across strides by weighted average of average variance per stride
                b_weighted_by_var = weighted_average(b, bvar, 0)
                w_weighted_by_var = weighted_average(w, bvar, 0)
                v_weighted_by_var = weighted_average(v, bvar, 0)

                # Append to output
                beta.append(b_weighted_by_var)
                eigenvalues.append(w_weighted_by_var)
                eigenvectors.append(v_weighted_by_var)

            else:
                # measuring with unfolded data (independent of layer type)
                ccov = torch.cov(input.T) # get covariance of input
                crank = torch.linalg.matrix_rank(ccov) # measure rank of covariance
                w, v = torch.linalg.eigh(ccov) # measure eigenvalues and vectors
                w_idx = torch.argsort(-w) # sort by eigenvalue highest to lowest
                w = w[w_idx]
                v = v[:,w_idx]

                # Automatically set eigenvalues to 0 when they are numerical errors!
                w[crank:] = 0

                # Measure abs value of dot product of weights on eigenvectors for each layer
                weight = weight / torch.norm(weight, dim=1, keepdim=True)
                beta.append(weight.cpu() @ v)

                # Append eigenvalues and eigenvectors to output
                eigenvalues.append(w)
                eigenvectors.append(v)

        return beta, eigenvalues, eigenvectors
    
    def measure_class_eigenfeatures(self, dataloader, eigenvectors, rms=False, DEVICE=None, with_updates=True):
        """
        propagate an entire dataset through the network and measure the contribution
        of each eigenvector to each element of the class

        to keep things in a useful tensor format, will match samples across classes 
        and therefore may ignore "extra" samples if the dataloader doesn't have equal
        representation across classes. Keep in mind it will use the first N samples 
        per class where N=min_samples_per_class, so using a random dataloader is smart.

        returns list of beta by class, where the list has len()==num_layers
        and each element is a tensor with size (num_classes, num_dimensions, num_samples_per_class)

        if rms=True, will convert beta_by_class to an average with the RMS method
        """
        # Set device automatically if not provided
        DEVICE = get_device(self)

        allinputs = []
        alllabels = []
        dataloop = tqdm(dataloader) if with_updates else dataloader
        for input, label in dataloop:
            input = input.to(DEVICE)
            alllabels.append(label)
            allinputs.append(self.get_layer_inputs(input, precomputed=False))

        # concatenate input to layers and labels into single tensors
        inputs_to_layers = [torch.cat([input[layer] for input in allinputs], dim=0).detach().cpu()
                            for layer in range(self.num_layers())]
        labels = torch.cat(alllabels)
        
        # get stacked indices to the elements of each class
        num_classes = len(dataloader.dataset.classes)
        idx_to_class = [torch.where(labels==ii)[0] for ii in range(num_classes)]
        num_per_class = [len(idx) for idx in idx_to_class]
        min_per_class = min(num_per_class)
        if any([npc>min_per_class for npc in num_per_class]):
            max_per_class = max(num_per_class)
            if (max_per_class / min_per_class) > 2:
                warn_message = f"Number of elements to each class is unequal (min={min_per_class}, max={max_per_class}). Clipping examples."
                warn(warn_message, RuntimeWarning, stacklevel=1)
            idx_to_class = [idx[:min_per_class] for idx in idx_to_class]

        # use single tensor for fast indexing
        idx_to_class = torch.stack(idx_to_class).unsqueeze(1)

        # measure the contribution of each eigenvector on the representation of each input
        beta_activity = [(input @ evec).T.unsqueeze(0) for input, evec in zip(inputs_to_layers, eigenvectors)]

        # organize activity by class in extra dimension
        beta_by_class = [torch.gather(betas.expand(num_classes, -1, -1), 2, idx_to_class.expand(-1, betas.size(1), -1)) for betas in beta_activity]

        # get average by class with RMS method (root-mean-square) if requested
        if rms:
            beta_by_class = [torch.sqrt(torch.mean(beta**2, dim=2)) for beta in beta_by_class]
        
        # return beta by class in requested format (average or not based on rms value)
        return beta_by_class
    

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


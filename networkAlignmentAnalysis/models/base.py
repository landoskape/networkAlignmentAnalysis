from tqdm import tqdm
from abc import ABC, abstractmethod
import torch
from torch import nn

from warnings import warn

from .layers import LAYER_REGISTRY, REGISTRY_REQUIREMENTS, check_metaparameters
from ..utils import check_iterable
from ..utils import get_maximum_strides
from ..utils import weighted_average
from ..utils import get_device
from ..utils import remove_by_idx
from ..utils import set_net_mode
from ..utils import get_unfold_params
from ..utils import smart_pca
from ..utils import alignment


class AttributeReference:
    """
    Simple class designed to be a reference to the parent class as an attribute.

    This is required for compatibility with using DDP for training pytorch modules,
    since we'll sometimes train DDP models and sometimes not, we want the code to
    work the same way. However, if you instantiate a DPP model from a network:

    net = AlignmentNetwork()
    ddp_net = DDP(net)

    Then the AlignmentNetwork methods will only be accessible in ddp_net.module.__. Therefore,
    if we have a system whereby the AlignmentNetwork methods can also be accessed in
    net.module.__, then the code can be the same regardless of whether we're using DDP or not.

    Usage
    -----
        net = AlignmentNetwork() (or any object instantiation)
        net.module = AttributeReference(net)
        ---or---
        class class_name:
            def __init__(self, *args, **kwargs):
                self.module = AttributeReference(self)
    """

    def __init__(self, parent):
        self.parent = parent

    def __getattr__(self, name):
        if hasattr(self.parent, name):
            return getattr(self.parent, name)
        else:
            raise AttributeError(f"parent object (instance of {type(self.parent)}) has no attribute '{name}'")


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

    def __init__(self, reference=True, **kwargs):
        super().__init__()  # register it as a nn.Module
        self.layers = nn.ModuleList()  # a list of all modules in the forward pass
        self.metaparameters = []  # list of dictionaries containing metaparameters for each layer
        self.hidden = []  # list of tensors containing hidden activations
        self.ignore_flag = kwargs.pop("ignore_flag", False)  # whether to ignore flagged layers
        self.initialize(**kwargs)  # initialize the architecture using child class method
        if reference:
            # create reference to self in "model" attribute for compatibility with DDP
            self.module = AttributeReference(self)

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

        Required kwargs are: name, layer_index, alignment_method, unfold, ignore, ...
        """
        if not isinstance(layer, nn.Module):
            raise TypeError(f"provided layer is of type: {type(layer)}, but only nn.Module objects are permitted!")

        # load default metaparameters for this type of layer (or empty dictionary)
        metaparameters = LAYER_REGISTRY.get(type(layer), {})

        # for each possible entry in layer metaparameters, check if it's provided and not none, then update it
        for metaprms in REGISTRY_REQUIREMENTS:
            if metaprms in kwargs and kwargs[metaprms] is not None:
                metaparameters[metaprms] = kwargs[metaprms]

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
        return not metaprms["ignore"] and not (metaprms["flag"] and self.ignore_flag)

    def set_ignore_flag(self, ignore_flag):
        """method for setting the ignore_flag switch"""
        assert isinstance(ignore_flag, bool), "ignore_flag setting must be a bool"
        self.ignore_flag = ignore_flag

    def forward(self, x, store_hidden=False):
        """standard forward pass of all layers with option of storing hidden activations (and output)"""
        self.hidden = []  # always reset so as to not keep a previous forward pass accidentally
        for layer in self.layers:
            x = layer(x)  # pass through next layer
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

        if hasattr(self, "dropout"):
            setattr(self, "dropout", p)

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
        if hasattr(self, "dropout"):
            setattr(self, "dropout", p)

    @torch.no_grad()
    def get_layer_inputs(self, x, precomputed=False, idx=None):
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
        if idx is None:
            return layer_inputs
        return layer_inputs[idx]

    @torch.no_grad()
    def get_layer_outputs(self, x=None, precomputed=False, idx=None):
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
        if idx is None:
            return layer_outputs
        return layer_outputs[idx]

    @torch.no_grad()
    def get_layer(self, layer, metaprms):
        """get alignment layer from **layer** (which might be a sequential layer, etc.) based on metaprms"""
        if metaprms["layer_index"] is None:
            return layer
        else:
            return layer[metaprms["layer_index"]]

    @torch.no_grad()
    def get_alignment_layers(self, idx=None):
        """convenience method for retrieving registered layers for alignment measurements throughout the network"""
        layers = []
        for layer, metaprms in zip(self.layers, self.metaparameters):
            if self._include_layer(metaprms):
                # ATL: deprecated line because the lambda method layer_handle prevented saving-- layers.append(metaprms["layer_handle"](layer))
                layers.append(self.get_layer(layer, metaprms))  # get layer without layer_handle lambda method
        if idx is None:
            return layers
        return layers[idx]

    @torch.no_grad()
    def get_alignment_metaparameters(self, idx=None):
        """convenience method for retrieving registered layers for alignment measurements throughout the network"""
        metaparameters = []
        for metaprms in self.metaparameters:
            if self._include_layer(metaprms):
                metaparameters.append(metaprms)
        if idx is None:
            return metaparameters
        return metaparameters[idx]

    @torch.no_grad()
    def get_alignment_layer_indices(self, idx=None):
        """convenience method for retrieving the absolute indices of alignment layers throughout the network"""
        idx_layers = []
        for ii, metaprms in enumerate(self.metaparameters):
            if self._include_layer(metaprms):
                idx_layers.append(ii)
        if idx is None:
            return idx_layers
        return idx_layers[idx]

    @torch.no_grad()
    def get_alignment_weights(self, idx=None, flatten=False):
        """
        convenience method for retrieving registered weights for alignment measurements throughout the network

        if flatten=True, will flatten weights so they have shape (nodes/channels, numel_per_weight)
        """
        # go through each layer and retrieve weight as desired
        weights = []
        for layer in self.get_alignment_layers():
            # get weight data for this layer
            weight = layer.weight.data

            # if requesting flat weights, flatten them
            if flatten:
                weight = weight.flatten(start_dim=1)

            # add weights to list
            weights.append(weight)

        # return
        if idx is None:
            return weights
        return weights[idx]

    def _preprocess_inputs(self, inputs_to_layers, compress_convolutional=True):
        """
        helper method for processing inputs to layers as needed for certain alignment operations

        Operations by layer type
        ------------------------
        linear layer:
            will leave inputs to layers unchanged if input to a feedforward layer
        convolutional layer:
            if compress_convolutional=True, will unfold inputs to (batch * num_strides, conv_weight_dim)
            otherwise, will unfold inputs to (batch, conv_weight_dim, num_strides)
        """
        # initialize new list of inputs to layers
        preprocessed = []

        # get layer parameters and metaparameters
        layers = self.get_alignment_layers()
        metaprms = self.get_alignment_metaparameters()

        # do requested processing and add to output
        for input, layer, metaprm in zip(inputs_to_layers, layers, metaprms):
            if metaprm["unfold"]:
                # if convolutional layer, unfold layer to (batch / conv_dim / num_strides)
                layer_prms = get_unfold_params(layer)
                unfolded_input = torch.nn.functional.unfold(input, layer.kernel_size, **layer_prms)
                if compress_convolutional:
                    unfolded_input = unfolded_input.transpose(1, 2).contiguous().view(-1, unfolded_input.size(1))
                preprocessed.append(unfolded_input)
            else:
                # if linear layer, no preprocessing should ever be required
                preprocessed.append(input)

        # return processed input data
        return preprocessed

    @torch.no_grad()
    def compare_weights(self, weights):
        current_weights = self.get_alignment_weights()
        delta_weights = []
        for iw, cw in zip(weights, current_weights):
            delta_weights.append(torch.norm(cw.flatten(1) - iw.flatten(1), dim=1))
        return delta_weights

    @torch.no_grad()
    def measure_alignment(self, x, precomputed=False, method="alignment"):
        # Pre-layer activations start with input (x) and ignore output
        inputs_to_layers = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(inputs_to_layers, compress_convolutional=True)
        weights = self.get_alignment_weights(flatten=True)
        return [alignment(input, weight, method=method) for input, weight in zip(preprocessed, weights)]

    @torch.no_grad()
    def forward_targeted_dropout(self, x, idxs, layers):
        """
        perform forward pass with targeted dropout on output of hidden layers

        **idxs** and **layers** are matched length tuples describing the layer to dropout
        in and the idxs in that layer to dropout. The dropout happens in the activations
        of the layer (so layer=(0) corresponds to the output of the first layer).

        returns the output accounting for targeted dropout and also the full list of hidden
        activations after targeted dropout
        """
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers need to be iterables with the same length"
        assert len(idxs) == len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers) == len(set(layers)), "layers must not have any repeated elements"
        assert all([layer >= 0 and layer < len(self.layers) - 1 for layer in layers]), "dropout only works on first N-1 layers"

        hidden_outputs = []
        for idx_layer, (layer, metaprms) in enumerate(zip(self.layers, self.metaparameters)):
            x = layer(x)  # pass through next layer
            if idx_layer in layers:
                dropout_idx = idxs[{val: idx for idx, val in enumerate(layers)}[idx_layer]]
                fraction_dropout = len(dropout_idx) / x.shape[1]
                x[:, dropout_idx] = 0
                x = x * (1 - fraction_dropout)
            if self._include_layer(metaprms):
                hidden_outputs.append(x)

        # return output of network and outputs of each alignment layer
        return x, hidden_outputs

    @torch.no_grad()
    def forward_eigenvector_dropout(self, x, eigenvalues, eigenvectors, idxs, layers):
        """
        perform forward pass with targeted dropout of loadings on eigenvectors on input to hidden layers

        **eigenvalues**, **eigenvectors**, **idxs** and **layers** are matched length tuples describing:
        eigenvalues: the eigenvalues of each eigenvector for the input to each layer
        eigenvectors: the eigenvectors of activity corresponding to the input to each layer
        idxs: which eigenvectors to dropout from the activity as it propagates through the network
        layers: which layers to do dropouts in (an index)

        for convolutional layers, dropout is done on each stride independently (with the same subspace)

        returns the output accounting for targeted dropout and also the full list of hidden
        activations after targeted dropout. will correct the norm based on the fraction of
        variance contained in the eigenvalues
        """
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers need to be iterables with the same length"
        assert len(idxs) == len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers) == len(set(layers)), "layers must not have any repeated elements"
        assert len(layers) == len(eigenvalues), "list of eigenvalues must have same length as list of layers"
        assert len(layers) == len(eigenvectors), "list of eigenvectors must have same length as list of layers"
        device = get_device(x)

        hidden_inputs = []
        for idx_layer, (layer, metaprms) in enumerate(zip(self.layers, self.metaparameters)):
            if idx_layer in layers:
                # we need to get the target subspace after dropping out eigenvectors

                # get index to target layer
                idx_to_layer = {val: idx for idx, val in enumerate(layers)}[idx_layer]

                # get dropout indices of which eigenvectors to remove
                dropout_idx = idxs[idx_to_layer]

                # retrieve only the requested eigenvectors & eigenvalues
                dropout_evec = remove_by_idx(eigenvectors[idx_to_layer].to(device), dropout_idx, 1)
                dropout_eval = remove_by_idx(eigenvalues[idx_to_layer].to(device), dropout_idx, 0)

                # correction is defined as the square root as the ratio of variance preserved in the subspace
                # this will roughly preserve the average norm of the data for each sample
                dropout_correction = torch.sqrt(torch.sum(eigenvalues[idx_to_layer]) / torch.sum(dropout_eval))

            else:
                # if not target layer, we don't want to do any subspace processing
                dropout_evec = None
                dropout_correction = None

            # do forward pass through this layer
            kwargs = dict(subspace=dropout_evec, correction=dropout_correction)
            x, input_to_layer = self._forward_subspace(x, layer, metaprms, **kwargs)

            if self._include_layer(metaprms):
                hidden_inputs.append(input_to_layer)

        # return output of network and inputs to each alignment layer
        return x, hidden_inputs

    def _forward_subspace(self, x, layer, metaprms, subspace=None, correction=None):
        """helper for sending to forward function of desired type"""
        if metaprms["unfold"]:
            return self._forward_subspace_convolutional(x, layer, metaprms, subspace=subspace, correction=correction)
        else:
            return self._forward_subspace_linear(x, layer, metaprms, subspace=subspace, correction=correction)

    def _forward_subspace_linear(self, x, layer, _, subspace=None, correction=None):
        """
        implement forward pass for linear layer with optional subspace projection of input to layer
        """
        if subspace is not None:
            x = torch.matmul(torch.matmul(x, subspace), subspace.T)
            if correction is not None:
                x = x * correction
        out = layer(x)
        return out, x

    def _forward_subspace_convolutional(self, x, layer, metaprms, subspace=None, correction=None):
        """
        implement forward pass for convolutional layer with optional subspace projection of input

        if subspace provided, will project x onto the subspace then onto it's transpose to keep
        only some dimensions of the activity while keeping x in the same basis.
        (e.g. new_x = subspace @ subspace.T @ x)

        then will pass through the layer.

        projects onto the subspace within each stride of the convolution
        """

        def _conv_with_subspace(x, layer, subspace, correction):
            """internal helper for convolving in a subspace"""
            # start by getting size of input to conv layer and layer parameters
            h_max, w_max = get_maximum_strides(x.size(2), x.size(3), layer)
            layer_prms = get_unfold_params(layer)

            # perform convolution in unfolded space
            weight = layer.weight.data
            weight = weight.view(weight.size(0), -1)

            # this is the layer we want to reimplement with a subspace projection
            x = torch.nn.functional.unfold(x, layer.kernel_size, **layer_prms)

            # project out subspace
            x = torch.matmul(subspace, torch.matmul(subspace.T, x))

            # apply multiplicative gain correction if provided
            if correction is not None:
                x = x * correction

            # save input to target conv layer
            input_to_conv = x.clone()

            # convolve
            x = torch.matmul(weight, x).view(x.size(0), weight.size(0), h_max, w_max)

            # add bias
            x = x + layer.bias.view(-1, 1, 1)

            return x, input_to_conv

        if subspace is not None:
            if isinstance(layer, torch.nn.Sequential):
                layer_idx = metaprms["layer_index"]
                for idx, sublayer in enumerate(layer):
                    if idx == layer_idx:
                        # if conv layer, do convolution with subspace
                        x, input_to_conv = _conv_with_subspace(x, sublayer, subspace, correction)
                    else:
                        # if not target layer, we can process with no funny business
                        x = sublayer(x)

                # return output of layer and input to convolutional layer
                return x, input_to_conv

            else:
                # if not packaged in sequential, can do this directly
                x, input_to_conv = _conv_with_subspace(x, layer, subspace, correction)

            return x, input_to_conv

        else:
            # if not using subspace, just pass input through layer and return input/output
            return layer(x), x

    @torch.no_grad()
    def measure_eigenfeatures(self, inputs, with_updates=True, centered=True):
        """
        measure the eigenvalues and eigenvectors of the input to each layer
        and also measure how much each weight array uses each eigenvector

        computing eigenfeatures is intensive for big matrices so it's not a
        good idea to this on unfolded data in convolutional layers. It may be
        a good idea to do it sometimes -- I think sklearn's IncrementalPCA
        algorithm is best for this. But it still takes a while so shouldn't be
        done frequently, only after training for important networks.

        if centered=True, will measure eigenfeatures of true covariance matrix.
        if centered=False, will measure eigenfeatures of uncentered X.T @ X where
        x is the input to each alignment layer.

        for convolutional layers, will unfold and measure eigenfeatures for each
        stride (and take the average across strides weighted by input variance)
        """
        # retrieve weights, reshape, and flatten inputs as required
        weights = self.get_alignment_weights(flatten=True)
        inputs = self._preprocess_inputs(inputs, compress_convolutional=True)

        # measure eigenfeatures
        return self._measure_layer_eigenfeatures(inputs, weights, centered=centered, with_updates=with_updates)

    def measure_class_eigenfeatures(self, inputs, labels, eigenvectors, rms=False, with_updates=True):
        """
        propagate an entire dataset through the network and measure the contribution
        of each eigenvector to each element of the class

        to keep things in a useful tensor format, will match samples across classes and
        therefore may ignore "extra" samples if the dataloader doesn't have equal
        representation across classes. Keep in mind it will use the first N samples per
        class where N=min_samples_per_class, so using a random dataloader is a good idea.

        returns list of beta by class, where the list has len()==num_layers
        and each element is a tensor with size (num_classes, num_dimensions, num_samples_per_class)

        if rms=True, will convert beta_by_class to an average with the RMS method
        """
        # get stacked indices to the elements of each class
        classes = torch.unique(labels)
        num_classes = len(classes)
        idx_to_class = [torch.where(labels == ii)[0] for ii in classes]
        num_per_class = [len(idx) for idx in idx_to_class]
        min_per_class = min(num_per_class)
        if any([npc > min_per_class for npc in num_per_class]):
            max_per_class = max(num_per_class)
            if (max_per_class / min_per_class) > 2:
                warn_message = f"Number of elements to each class is unequal (min={min_per_class}, max={max_per_class}). Clipping examples."
                warn(warn_message, RuntimeWarning, stacklevel=1)
            idx_to_class = [idx[:min_per_class] for idx in idx_to_class]

        # use single tensor for fast indexing
        idx_to_class = torch.stack(idx_to_class).unsqueeze(1)

        # measure the contribution of each eigenvector on the representation of each input
        beta_activity = []
        inputs = self._preprocess_inputs(inputs, compress_convolutional=False)
        zipped = zip(inputs, eigenvectors, self.get_alignment_layers(), self.get_alignment_metaparameters())
        for input, evec, layer, metaprm in zipped:
            if metaprm["unfold"]:
                print("measure_class_eigenfeatures has not integrated new convolutional approach")
                stride_var = torch.var(input, dim=1, keepdim=True)
                projection = torch.matmul(evec.T, input)
                projection = weighted_average(projection, stride_var, dim=2)
                beta_activity.append(projection.T.unsqueeze(0))
            else:
                beta_activity.append((input @ evec).T.unsqueeze(0))

        # organize activity by class in extra dimension
        beta_by_class = [torch.gather(betas.expand(num_classes, -1, -1), 2, idx_to_class.expand(-1, betas.size(1), -1)) for betas in beta_activity]

        # get average by class with RMS method (root-mean-square) if requested
        if rms:
            beta_by_class = [torch.sqrt(torch.mean(beta**2, dim=2)) for beta in beta_by_class]

        # return beta by class in requested format (average or not based on rms value)
        return beta_by_class

    def _measure_layer_eigenfeatures(self, inputs, weights, centered=True, with_updates=True):
        """
        helper method for measuring eigenfeatures of each layer

        input should be preprocessed weights (see _preprocess_inputs()) using compress_convolutional=True
        weights should be preprocessed weights (in the case of convolutional layers, see get_alignment_weights())
        """
        beta, eigenvalues, eigenvectors = [], [], []

        # go through each layers inputs, weights, and metaparameters
        zipped = enumerate(zip(inputs, weights))
        iterate = tqdm(zipped) if with_updates else zipped
        for ii, (input, weight) in iterate:
            """
            #ATL 240227: this used to be divided into linear vs. convolutional
            but now _preprocess_inputs will fold stride dimension in with batch dimension
            so it will behave the same way as a linear layer
            """
            # measure evals and evecs across input
            w, v = smart_pca(input.T, centered=centered)

            # Measure abs value of dot product of weights on eigenvectors for each layer
            weight = weight / torch.norm(weight, dim=1, keepdim=True)
            beta.append(weight.cpu() @ v)

            # Append eigenvalues and eigenvectors to output
            eigenvalues.append(w)
            eigenvectors.append(v)

            """
            #ATL 240227 - obsolete code now that stride dimension is being folded into batch dimension
            # if a convolutional layer, then:
            if metaprm['unfold']:
                # measure variance across dimensions (the actual variance within each stride)
                # then take average across batch
                bvar = torch.mean(torch.var(input, dim=1), dim=0) 
                
                # get eigenvalues and eigenvectors for each stride (treat stride as batch dimension here)
                w, v = smart_pca(input.permute((2, 1, 0)), centered=centered)
                
                # Measure abs value of dot product of weights on eigenvectors for each layer
                num_strides = v.size(0)
                weight = weight / torch.norm(weight, dim=1, keepdim=True)
                b = torch.bmm(weight.cpu().unsqueeze(0).expand(num_strides, -1, -1), v)

                # Contract across strides by weighted average of average variance per stride
                b_weighted_by_var = weighted_average(b, bvar.view(-1, 1, 1), 0)
                w_weighted_by_var = weighted_average(w, bvar.view(-1, 1), 0)
                v_weighted_by_var = weighted_average(v, bvar.view(-1, 1, 1), 0)

                # Append to output
                beta.append(b_weighted_by_var)
                eigenvalues.append(w_weighted_by_var)
                eigenvectors.append(v_weighted_by_var)
            """

        return beta, eigenvalues, eigenvectors

    def _process_collect_activity(self, dataset, train_set=True, with_updates=True, use_training_mode=False):
        """
        helper for processing and collecting activity of network in response to all inputs of dataloader

        automatically places all data on cpu

        returns inputs to each alignment layer, concatenated across entire dataloader as a per layer list
        returns labels of entire dataset

        with_updates turns on or off the progress bar (using tqdm)
        if use_training_mode=False, will put net into evaluation mode (and return to original mode)
        if use_training_mode=True, will put net into training mode (and return to original mode)
        """
        # get device of network
        device = get_device(self)

        # put network in evaluation mode
        training_mode = set_net_mode(self, training=use_training_mode)

        # store input and measure activations for every element in dataloader
        allinputs = []
        alllabels = []
        dataloader = dataset.train_loader if train_set else dataset.test_loader
        dataloop = tqdm(dataloader) if with_updates else dataloader
        for batch in dataloop:
            input, labels = dataset.unwrap_batch(batch, device=device)
            layer_inputs = [input.cpu() for input in self.get_layer_inputs(input, precomputed=False)]
            allinputs.append(layer_inputs)
            alllabels.append(labels.cpu())

        # return network to original training/eval mode, whatever it was
        set_net_mode(self, training=training_mode)

        # create large list of tensors containing input to each layer
        inputs = [torch.cat([input[layer] for input in allinputs], dim=0) for layer in range(self.num_layers())]
        labels = torch.cat(alllabels, dim=0)

        # return outputs
        return inputs, labels

    @torch.no_grad()
    def shape_eigenfeatures(self, idx_layers, eigenvalues, eigenvectors, eval_transform):
        """
        method for shaping the eigenfeatures of a network

        use eval_transform to shape a network by changing the scale of each
        eigenvector's contribution to the weights based on the associated
        eigenvalue for a specific set of layers.

        idx_layers is a list indicating which layers to shape (where the indices
        should correspond to indices in self.get_alignment_layer_indices())

        eigenvalues and eigenvectors should be a list with length=len(idx_layers)
        and each should correspond to the eigenvalues & eigenvectors of the input
        to each layer in idx_layers

        eval_transform is a callable function that takes a set of eigenvalues and
        returns the desired scale of eigenvectors associated with each eigenvalue
        for example, if eigenvalues[0]=[1, 0.5, 0.25, 0.125]*37.9991, eval_transform
        might return [1, 1, 1, 0] which simply "kills" the last eigenvector
        alternatively, it could return [1, 0.25, 0.25**2, 0.125**2]*37.9991**2/sum
        where it shapes each eigenvector by the square of the eigenvalues
        """
        # do some input checks
        assert all([idx in self.get_alignment_layer_indices() for idx in idx_layers]), (
            "idx_layers includes some indices not in alignment layers",
            f"(provided: {idx_layers}, alignment_layer_indices: {self.get_alignment_layer_indices()})",
        )
        assert len(idx_layers) == len(eigenvalues), "length of idx_layers and eigenvalues doesn't match"
        assert len(idx_layers) == len(eigenvectors), "length of idx_layers and eigenvectors doesn't match"

        # make sure eigenvalues and eigenvalues are on same device as network
        device = get_device(self)
        eigenvalues = [evals.to(device) for evals in eigenvalues]
        eigenvectors = [evecs.to(device) for evecs in eigenvectors]

        # get weights and original shapes of requested alignment layers
        weight_shape = [self.get_alignment_weights(idx=idx).shape for idx in idx_layers]
        weights = [self.get_alignment_weights(idx=idx, flatten=True) for idx in idx_layers]

        # measure original norm of weights
        norm_of_weights = [torch.norm(weight, dim=1, keepdim=True) for weight in weights]

        # normalize weight vector
        weights = [weight / torch.norm(weight, dim=1, keepdim=True) for weight in weights]

        # for each layer, process the eigenvalues, shape the weights, and update the network
        zipped = zip(idx_layers, eigenvalues, eigenvectors, weights, norm_of_weights, weight_shape)
        for idx, evals, evecs, weight, norm_weight, shape in zipped:
            # transform eigenvalues
            eval_keep_fraction = eval_transform(evals)
            assert (
                type(eval_keep_fraction) == type(evals) and eval_keep_fraction.shape == evals.shape
            ), "eval_transform returned new evals with the wrong type or shape"
            # define a projection matrix that scales the contribution of each eigenvalue by eval_keep_fraction
            proj_matrix = evecs @ torch.diag(eval_keep_fraction) @ evecs.T
            # shape the weights
            shaped_weights = weight @ proj_matrix
            # renormalize them to their original norm
            shaped_weights = shaped_weights / torch.norm(shaped_weights, dim=1, keepdim=True)  # normalize
            shaped_weights = shaped_weights * norm_weight
            # reshape to original shape
            shaped_weights = torch.reshape(shaped_weights, shape)
            # update the network
            self.get_alignment_layers(idx=idx).weight.data = shaped_weights


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

#

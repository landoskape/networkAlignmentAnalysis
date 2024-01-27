from typing import List
from contextlib import contextmanager
from functools import wraps
import numpy as np
from scipy.linalg import null_space
from sklearn.decomposition import IncrementalPCA
import torch
from matplotlib import pyplot as plt
from torchvision import transforms


# -- deprecation block --
from warnings import warn

def avg_align_by_layer(full):
    warn("avg_align_by_layer is deprecated, change to avg_value_by_layer!", DeprecationWarning, stacklevel=2)
    return avg_value_by_layer(full)

def align_by_layer(full, layer):
    warn("align_by_layer is deprecated, change to value_by_layer!", DeprecationWarning, stacklevel=2)
    return value_by_layer(full, layer)
# ------------------------


# -------------- context managers & decorators --------------
@contextmanager
def no_grad(no_grad=True):
    if no_grad:
        with torch.no_grad():
            yield
    else:
        yield

def test_nets(func):
    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        # get original training mode and set to eval
        in_training_mode = [set_net_mode(net, training=False) for net in nets]

        # do decorated function
        func_outputs = func(nets, *args, **kwargs)

        # return networks to whatever mode they used to be in 
        for train_mode, net in zip(in_training_mode, nets):
            set_net_mode(net, training=train_mode)

        # return decorated function outputs
        return func_outputs
    
    # return decorated function
    return wrapper

def train_nets(func):
    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        # get original training mode and set to train
        in_training_mode = [set_net_mode(net, training=True) for net in nets]

        # do decorated function
        func_outputs = func(nets, *args, **kwargs)

        # return networks to whatever mode they used to be in 
        for train_mode, net in zip(in_training_mode, nets):
            set_net_mode(net, training=train_mode)

        # return decorated function outputs
        return func_outputs
    
    # return decorated function
    return wrapper

def set_net_mode(net, training=True):
    """helper for setting mode of network and returning current mode"""
    # get current mode of network
    in_training_mode = net.training
    # set to training mode or evaluation mode
    if training:
        net.train()
    else:
        net.eval()
    # return original mode of network
    return in_training_mode

# ------- some other things -------
def get_device(obj):
    """simple method to get device of input tensor or nn.Module"""
    if isinstance(obj, torch.nn.Module):
        return next(obj.parameters()).device.type
    elif isinstance(obj, torch.Tensor):
        return 'cuda' if obj.is_cuda else 'cpu'
    else:
        raise ValueError("")
    
def check_iterable(val):
    """duck-type check if val is iterable"""
    try:
        _ = iter(val)
    except:
        return False
    else:
        return True

def remove_by_idx(input, idx, dim):
    """
    remove part of input indexed by idx on dim
    """
    idx_keep = [i for i in range(input.size(dim)) if i not in idx]
    return torch.index_select(input, dim, torch.tensor(idx_keep).to(input.device))

def smartcorr(input):
    """
    Performs torch corrcoef on the input data but sets each pair-wise correlation coefficent
    to 0 where the activity has no variance (var=0) for a particular dimension (replaces nans with zeros)sss
    """
    idx_zeros = torch.var(input, dim=1)==0
    cc = torch.corrcoef(input)
    cc[idx_zeros,:] = 0
    cc[:,idx_zeros] = 0
    return cc

def batch_cov(input, centered=True):
    """
    Performs batched covariance on input data of shape (batch, dim, samples) or (dim, samples)

    Where the resulting batch covariance matrix has shape (batch, dim, dim) or (dim, dim)
    and bcov[i] = torch.cov(input[i]) if input.ndim==3

    if centered=True (default) will subtract the means first
    """
    assert (input.ndim == 2) or (input.ndim == 3), "input must be a 2D or 3D tensor"
    # check if batch dimension was provided
    no_batch = input.ndim == 2 
    
    # add an empty batch dimension if not provided
    if no_batch: 
        input = input.unsqueeze(0) 
    
    # measure number of samples of each input matrix
    S = input.size(2) 
    
    # subtract mean if doing centered covariance
    if centered:
        input = input - input.mean(dim=2, keepdim=True) 

    # measure covariance of each input matrix
    bcov = torch.bmm(input, input.transpose(1, 2))
    
    # correct for number of samples
    bcov /= (S-1)
    
    # remove empty batch dimension if not provided
    if no_batch: 
        bcov = bcov.squeeze(0) 

    return bcov

def sklearn_pca(input, use_rank=True):
    """
    sklearn incrementalPCA algorithm serving as a replacement for eigh when it fails
    
    input should be a tensor with shape (num_samples, num_features) or it can be a 
    covariance matrix with (num_features, num_features)

    if use_rank=True, will set num_components to the rank of input and then fill out the
    rest of the components with random orthogonal components in the null space of the true
    components and set the eigenvalues to 0

    if use_rank=False, will attempt to fit all the components

    returns w, v where w is eigenvalues and v is eigenvectors sorted from highest to lowest
    """
    # dimension
    num_samples, num_features = input.shape

    # measure rank (or set to None)
    rank = int(torch.linalg.matrix_rank(input)) if use_rank else None

    # create and fit IncrementalPCA object on input data
    ipca = IncrementalPCA(n_components=rank).fit(input)

    # eigenvectors are the components
    v = ipca.components_

    # eigenvalues are the scaled singular values
    w = ipca.singular_values_**2 / num_samples

    # if v is a subspace of input (e.g. not a full basis, fill it out)
    if v.shape[0] < num_features:
        msg = "adding this because I think it should always be true, and if not I want to find out"
        assert w.shape[0] == v.shape[0], msg
        v_kernel = null_space(v).T
        v = np.vstack((v, v_kernel))
        w = np.concatenate((w, np.zeros(v_kernel.shape[0])))
    
    return torch.tensor(w, dtype=torch.float), torch.tensor(v, dtype=torch.float).T


# ------------------ alignment functions ----------------------
def alignment(input, weight, method='alignment'):
    """
    measure alignment (proportion variance explained) between **input** and **weight**
    
    computes the rayleigh quotient between each weight vector in **weight** and the **input** fed 
    into **weight**. Typically, **input** is the output in Layer L-1 and **weight** is from Layer L

    the output is normalized by the total variance in output of layer L-1 to measure the proportion 
    of variance of in **input** is explained by a projection onto node's weights in **weight**

    args
    ----
        input: (batch, neurons) torch tensor 
            - represents input activity being fed in to network weight layer
        weight: (num_out, num_in) torch tensor 
            - represents weights multiplied by input layer
        method: string, default='alignment'
            - which method to use to measure structure in **input** 
            - if 'alignment', uses covariance matrix of **input**
            - if 'similarity', uses correlation matrix of **input**

    returns
    -------
        alignment: (num_out, ) torch tensor
            - proportion of variance explained by projection of **input** onto each **weight** vector
    """
    assert method=='alignment' or method=='similarity', "method must be set to either 'alignment' or 'similarity' (or None, default is alignment)"
    if method=='alignment':
        cc = torch.cov(input.T)
    elif method=='similarity':
        cc = smartcorr(input.T)
    else: 
        raise ValueError(f"did not recognize method ({method}), must be 'alignment' or 'similarity'")
    # Compute rayleigh quotient
    rq = torch.sum(torch.matmul(weight, cc) * weight, axis=1) / torch.sum(weight * weight, axis=1)
    # proportion of variance explained by a projection of the input onto each weight
    return rq/torch.trace(cc)


def alignment_linear(activity, layer, method='alignment', **kwargs):
    """wrapper for alignment of linear layer, kwargs for compatibility"""
    return alignment(activity, layer.weight.data, method=method)


def alignment_convolutional(activity, layer, by_stride=True, method='alignment', **kwargs):
    """
    wrapper for alignment of convolutional layer (for conv2d)

    there are two natural methods - one is to measure the alignment using each 
    convolutional stride, the second is to measure the alignment of the full 
    unfolded layer as if it was a matrix multiplication. by_stride determines
    which one to use. 
    
    when by_stride=True, a weighted average of the alignment at each stride is
    taken where the weights are the variance in the data at that stride. This way,
    the output is a (num_output_channels, ) shaped tensor regardless of the setting
    used for by_stride.

    **kwargs is just for compatibility and accepting irrelevant arguments without breaking
    """
    h_max, w_max = get_maximum_strides(activity.shape[2], activity.shape[3], layer)
    if by_stride:
        preprocess = transforms.Pad(layer.padding)
        processed_activity = preprocess(activity)
        num_looks = h_max * w_max
        num_channels = layer.out_channels
        w_idx, h_idx = torch.meshgrid(torch.arange(0, layer.kernel_size[0]), torch.arange(0, layer.kernel_size[1]), indexing='xy')
        align_layer = torch.zeros((num_channels, num_looks))
        variance_stride = torch.zeros(num_looks)
        for h in range(h_max):
            for w in range(w_max):
                out_tuple = alignment_conv_look(processed_activity, layer, (h, w), (h_idx, w_idx), method=method)
                align_layer[:, w + h_max*h], variance_stride[w + h_max*h] = out_tuple
        
        # weighted average over variance_stride...
        align = torch.sum(align_layer * variance_stride.view(1, -1)) / torch.sum(variance_stride)
        return align
    else:
        layer_prms = get_unfold_params(layer)
        unfolded_input = torch.nn.functional.unfold(activity, layer.kernel_size, **layer_prms)
        unfolded_input = unfolded_input.transpose(1, 2).reshape(activity.size(0), -1)
        unfolded_weight = layer.weight.data.view(layer.weight.size(0), -1).repeat(1, h_max*w_max)
        return alignment(unfolded_input, unfolded_weight, method=method)

def alignment_conv_look(processed_activity, layer, stride, grid, method='alignment'):
    # Take (NI, C, H, W) (preprocessed) input activity
    # And compute the similarity for one "look" e.g. one position of the convolutional filter
    num_images = processed_activity.shape[0]
    num_elements = processed_activity.shape[1] * layer.kernel_size[0] * layer.kernel_size[1]
    h_idx = grid[0] + stride[0]*layer.stride[0]
    w_idx = grid[1] + stride[1]*layer.stride[1]
    aligned_input = processed_activity[:, :, h_idx, w_idx].reshape(num_images, num_elements)
    aligned_weights = layer.weight.data.reshape(layer.out_channels, num_elements).detach()
    stride_variance = torch.mean(torch.var(aligned_input, dim=1))
    return alignment(aligned_input, aligned_weights, method=method), stride_variance

def get_maximum_strides(h_input, w_input, layer):
    h_max = int(np.floor((h_input + 2*layer.padding[0] - layer.dilation[0]*(layer.kernel_size[0] - 1) -1)/layer.stride[0] + 1))
    w_max = int(np.floor((w_input + 2*layer.padding[1] - layer.dilation[1]*(layer.kernel_size[1] - 1) -1)/layer.stride[1] + 1))
    return h_max, w_max

def get_unfold_params(layer):
    return dict(stride=layer.stride, padding=layer.padding, dilation=layer.dilation)

def correlation(output, method='corr'):
    """
    Expects a batch x neuron tensor of output activity of a layer

    Returns: 
    Pairwise variance or correlation coefficient between neurons across batch dimension

    NOTE:
    As of now, most models have processing layers after the alignment layer in each
    registered layer. That means that the data which will be sent to this method is
    not really the post-alignment layer activations... should change the models before
    using this correlation method much. 
    """
    if method=='var':
        return torch.cov(output.T)
    elif method=='corr':
        return smartcorr(output.T)
    else:
        raise ValueError(f"Method ({method}) not recognized, must be 'var' or 'corr'")

def correlation_linear(output, method='corr', **kwargs):
    """wrapper for correlation of linear layer, kwargs for compatibility"""
    return correlation(output, method=method)

def correlation_convolutional(output, method='corr', by_stride=False, **kwargs):
    """
    wrapper for correlation of convolutional layer (conv2d)
    
    there are two natural methods - one is to measure the correlation using each 
    convolutional stride, the second is to measure the correlation of the full 
    unfolded layer. by_stride determines which one to use. 
    
    when by_stride=True, a weighted average of the correlation at each stride is
    taken where the weights are the variance in the output (across channels and batch)
    at that stride. This way, the output is a (num_output_channels, num_output_channels)
    shaped tensor regardless of the setting used for by_stride.

    NOTE: 
    This choice for by_stride=True was made because it works and is a simple way
    of averaging across strides. There might be a smarter way to average where the per
    channel variance is used rather than the across channel variance. We should think
    about that and make a better decision, or potentially created an option for doing
    it either way depending on the scientific goal.

    kwargs for compatibility
    """
    if by_stride:
        num_channels, h_max, w_max = output.size(1), output.size(2), output.size(3)
        corr = torch.zeros((num_channels, num_channels), dtype=output.dtype).to(get_device(output))
        vars = []
        for h in range(h_max):
            for w in range(w_max):
                cvar = torch.var(output[:, :, h, w].flatten())
                vars.append(cvar)
                corr += correlation(output[:, :, h, w], method=method) * cvar
        corr /= torch.sum(torch.tensor(vars))
        return corr
    else:
        output_by_channel = output.transpose(0, 1).reshape(output.shape[1], -1) # channels x (batch*H*W)
        return correlation(output_by_channel.T, method=method)

def avg_value_by_layer(full):
    """
    return average value per layer across training

    **full** is a list of lists where the outer list is each snapshot through training or 
    minibatch etc and each inner list is the value for each node in the network across layers
    of a particular measurement
    
    For example:
    num_epochs = 1000
    nodes_per_layer = [50, 40, 30, 20]
    len(full) == 1000
    len(full[i]) == 4 ... for all i
    [f.shape for f in full[i]] = [50, 40, 30, 20] ... for all i

    this method will return a tensor of size (num_layers, num_epochs) of the average value (for
    whatever value is in **full**) for each list/list
    """
    num_epochs = len(full)
    num_layers = len(full[0])
    avg_full = torch.zeros((num_layers,num_epochs))
    for layer in range(num_layers):
        avg_full[layer,:] = torch.tensor([torch.mean(f[layer]) for f in full])
    return avg_full.cpu()

def value_by_layer(full: List[List[torch.Tensor]], layer: int) -> torch.Tensor:
    """
    return all value measurements for a particular layer from **full**

    **full** is a list of lists where the outer list is each snapshot through training or
    minibatch etc and each inner list is the value for each node in the network across layers

    this method will return just the part of **full** corresponding to the layer indexed
    by **layer** as a tensor of shape (num_epochs, num_nodes)

    see ``avg_value_by_layer`` for a little more explanation
    """
    return torch.cat([f[layer].view(1, -1) for f in full], dim=0).cpu()

def condense_values(full: List[List[List[torch.Tensor]]]) -> List[torch.Tensor]:
    """
    condense List[List[List[Tensor]]] representing some value measured across networks, batches, and layers, for each node in the layer

    returns list of #=num_layers tensors, where each tensor has shape (num_networks, num_batches, num_nodes_per_layer)

    full should be a list of list of lists
    the first list should have length = number of networks
    the second list should have length = number of batches
    the third list should have length = number of layers in the network (this has to be the same for each network!)
    the tensor should have shape = number of nodes in this layer (also must be the same for each network) (or can be anything as long as consistent across layers)
    """
    num_layers = len(full[0][0])
    return [torch.stack([value_by_layer(value, layer) for value in full]) for layer in range(num_layers)]

def transpose_list(list_of_lists):
    """helper function for transposing the order of a list of lists"""
    return list(map(list, zip(*list_of_lists)))

def named_transpose(list_of_lists):
    """
    helper function for transposing lists without forcing the output to be a list like transpose_list

    for example, if list_of_lists contains 10 copies of lists that each have 3 iterable elements you
    want to name "A", "B", and "C", then write:
    A, B, C = named_transpose(list_of_lists)
    """
    return map(list, zip(*list_of_lists))

def ptp(tensor, dim=None, keepdim=False):
    """
    simple method for measuring range of tensor on requested dimension or on all data
    """
    if dim is None:
        return tensor.max() - tensor.min()
    return tensor.max(dim, keepdim).values - tensor.min(dim, keepdim).values

def rms(tensor, dim=None, keepdim=False):
    """simple method for measuring root-mean-square on requested dimension or on all data in tensor"""
    if dim is None:
        return torch.sqrt(torch.mean(tensor**2))
    return torch.sqrt(torch.mean(tensor**2, dim=dim, keepdim=keepdim))

def compute_stats_by_type(tensor, num_types, dim, method='var'):
    """
    helper method for returning the mean and variance across a certain dimension
    where multiple types are concatenated on that dimension

    for example, suppose we trained 2 networks each with 3 sets of parameters
    and concatenated the loss in a tensor like [set1-loss-net1, set1-loss-net2, set2-loss-net1, ...]
    then this would contract across the nets from each set and return the mean and variance
    """
    num_on_dim = tensor.size(dim)
    num_per_type = int(num_on_dim / num_types)
    tensor_by_type = tensor.unsqueeze(dim)
    expand_shape = list(tensor_by_type.shape)
    expand_shape[dim+1] = num_per_type
    expand_shape[dim] = num_types
    tensor_by_type = tensor_by_type.view(expand_shape)
    type_means = torch.mean(tensor_by_type, dim=dim+1)
    if method=='var':
        type_dev = torch.var(tensor_by_type, dim=dim+1)
    elif method=='std':
        type_dev = torch.std(tensor_by_type, dim=dim+1)
    elif method=='se':
        type_dev = torch.std(tensor_by_type, dim=dim+1) / np.sqrt(num_per_type)
    elif method=='range':
        type_dev = ptp(tensor_by_type, dim=dim+1)
    else:
        raise ValueError(f"Method ({method}) not recognized.")

    return type_means, type_dev

def weighted_average(data, weights, dim, keepdim=False):
    """
    take the weighted average of **data** on a certain dimension with **weights**

    weights should be a nonnegative vector with the same size as data.size(dim)
    uses the standard formula:
    avg = data_i * weight_i
    """
    assert weights.ndim==1, "weights must be a 1-d tensor"
    data_ndim = data.ndim
    weight_dim = weights.size(0)
    new_view = [weight_dim if ii==dim else 1 for ii in range(data_ndim)]
    weights = weights.view(new_view)
    numerator = torch.sum(data * weights, dim=dim, keepdim=keepdim)
    denominator = torch.sum(weights, dim=dim, keepdim=keepdim)
    return numerator / denominator

def plot_rf(rf, width, alignment=None, alignBounds=None, showRFs=None, figSize=5):
    if showRFs is not None: 
        rf = rf.reshape(rf.shape[0], -1)
        idxRandom = np.random.choice(range(rf.shape[0]),showRFs,replace=False)
        rf = rf[idxRandom,:]
    else: 
        showRFs = rf.shape[0]
    # normalize
    rf = rf.T / np.abs(rf).max(axis=1)
    rf = rf.T
    rf = rf.reshape(showRFs, width, width)
    # If necessary, create colormap
    if alignment is not None:
        cmap = cm.get_cmap('rainbow', rf.shape[0])
        cmapPeak = lambda x : cmap(x)
        if alignBounds is not None:
            alignment = alignment - alignBounds[0]
            alignment = alignment / (alignBounds[1] - alignBounds[0])
        else:
            alignment = (alignment - alignment.min())
            alignment = alignment / alignment.max()
        
    # plotting
    n = int(np.ceil(np.sqrt(rf.shape[0])))
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    fig.set_size_inches(figSize,figSize)
    
    N = 1000
    for i in tqdm(range(rf.shape[0])):
        ax = axes[i // n][i % n]
        if alignment is not None:
            vals = np.ones((N, 4))
            cAlignment = alignment[i].numpy()
            cPeak = cmapPeak(alignment[i].numpy())
            vals[:, 0] = np.linspace(0, cPeak[0], N)
            vals[:, 1] = np.linspace(0, cPeak[1], N)
            vals[:, 2] = np.linspace(0, cPeak[2], N)
            usecmap = ListedColormap(vals)
            ax.imshow(rf[i], cmap=usecmap, vmin=-1, vmax=1)
        else:
            ax.imshow(rf[i], cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    for j in range(rf.shape[0], n * n):
        ax = axes[j // n][j % n]
        ax.imshow(np.ones_like(rf[0]) * -1, cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig
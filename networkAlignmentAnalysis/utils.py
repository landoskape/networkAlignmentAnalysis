import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
    
def check_iterable(val):
    """duck-type check if val is iterable"""
    try:
        _ = iter(val)
    except:
        return False
    else:
        return True
    
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
        idx_no_activity = torch.where(torch.std(input,axis=0)==0)[0]
        cc = torch.corrcoef(input.T)
        cc[idx_no_activity,:] = 0
        cc[:,idx_no_activity] = 0
    else: 
        raise ValueError(f"did not recognize method ({method}), must be 'alignment' or 'similarity'")
    # Compute rayleigh quotient
    rq = torch.sum(torch.matmul(weight, cc) * weight, axis=1) / torch.sum(weight * weight, axis=1)
    # proportion of variance explained by a projection of the input onto each weight
    return rq/torch.trace(cc)


def alignment_linear(activity, layer, method='alignment'):
    """wrapper for alignment of linear layer"""
    return alignment(activity, layer.weight.data, method=method)


def alignment_convolutional(activity, layer, each_stride=True, method='alignment'):
    """
    wrapper for alignment of convolutional layer (for conv2d)

    there are two natural methods - one is to measure the alignment using each 
    convolutional stride, the second is to measure the alignment of the full 
    unfolded layer as if it was a matrix multiplication. each_stride determines
    which one to use. 
    
    when each_stride=True, a weighted average of the alignment at each stride is
    taken where the weights are the variance in the data at that stride. This way,
    the output is a (num_output_channels, ) shaped tensor regardless of the setting
    used for each_stride.
    """
    h_max, w_max = get_maximum_strides(activity.shape[2], activity.shape[3], layer)
    if each_stride:
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
        layer_prms = dict(stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
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

def avg_from_full(full):
    """
    return average alignment per layer across training

    **full** is a list of lists where the outer list is each snapshot through training or 
    minibatch etc and each inner list is the alignment for each node in the network across layers
    
    For example:
    num_epochs = 1000
    nodes_per_layer = [50, 40, 30, 20]
    len(full) == 1000
    len(full[i]) == 4 ... for all i
    [f.shape for f in full[i]] = [50, 40, 30, 20] ... for all i

    this method will return a tensor of size (num_layers, num_epochs) of the average alignment (or
    whatever value is in **full**) for each list/list
    """
    num_epochs = len(full)
    num_layers = len(full[0])
    avg_full = torch.zeros((num_layers,num_epochs))
    for layer in range(num_layers):
        avg_full[layer,:] = torch.tensor([torch.mean(f[layer]) for f in full])
    return avg_full.cpu()

def layer_from_full(full, layer):
    """
    return all alignment measurements for a particular layer from **full**

    **full** is a list of lists where the outer list is each snapshot through training or
    minibatch etc and each inner list is the alignment for each node in the network across layers

    this method will return just the part of **full** corresponding to the layer indexed
    by **layer** as a tensor of shape (num_nodes, num_epochs)

    see ``avg_from_full`` for a little more explanation
    """
    return torch.cat([f[layer].view(-1, 1) for f in full], dim=1).cpu()

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

def _ptp(tensor, dim=None, keepdim=False):
    if dim is None:
        return tensor.max() - tensor.min()
    return tensor.max(dim, keepdim).values - tensor.min(dim, keepdim).singular_values_

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
        type_dev = _ptp(tensor_by_type, dim=dim+1)
    else:
        raise ValueError(f"Method ({method}) not recognized.")

    return type_means, type_dev



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
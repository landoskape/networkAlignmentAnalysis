import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
    

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

    if using each_stride=True, the output is a (out_channels, num_strides) tensor
    otherwise it is a (out_channels, ) tensor
    """
    h_max, w_max = get_maximum_strides(activity.shape[2], activity.shape[3], layer)
    if each_stride:
        preprocess = transforms.Pad(layer.padding)
        processed_activity = preprocess(activity)
        num_looks = h_max * w_max
        num_channels = layer.out_channels
        w_idx, h_idx = torch.meshgrid(torch.arange(0, layer.kernel_size[0]), torch.arange(0, layer.kernel_size[1]), indexing='xy')
        align_layer = torch.zeros((num_channels, num_looks))
        for h in range(h_max):
            for w in range(w_max):
                align_layer[:, w + h_max*h] = alignment_conv_look(processed_activity, layer, (h, w), (h_idx, w_idx), method=method)
        return align_layer
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
    alignedInput = processed_activity[:, :, h_idx, w_idx].reshape(num_images, num_elements)
    alignedWeights = layer.weight.data.reshape(layer.out_channels, num_elements).detach()
    return alignment(alignedInput, alignedWeights, method=method)

def get_maximum_strides(h_input, w_input, layer):
    h_max = int(np.floor((h_input + 2*layer.padding[0] - layer.dilation[0]*(layer.kernel_size[0] - 1) -1)/layer.stride[0] + 1))
    w_max = int(np.floor((w_input + 2*layer.padding[1] - layer.dilation[1]*(layer.kernel_size[1] - 1) -1)/layer.stride[1] + 1))
    return h_max, w_max

def avg_from_full(full):
    num_epochs = len(full)
    num_layers = len(full[0])
    avg_full = torch.zeros((num_layers,num_epochs))
    for layer in range(num_layers):
        avg_full[layer,:] = torch.tensor([torch.nanmean(f[layer]) for f in full])
    return avg_full.cpu()

def layer_from_full(full,layer,dim=1):
    if dim==1: 
        return torch.cat([f[layer][:,None] for f in full],dim=dim).cpu() 
    elif dim==2:
        return torch.cat([f[layer][:,:,None] for f in full],dim=dim).cpu() 
    else:
        raise ValueError("Haven't coded layer_from_full for dimensions other than 1 or 2!")


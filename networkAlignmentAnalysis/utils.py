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
    b,n = input.shape
    m = weight.shape[1]
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
    return alignment(activity, layer.weight.data, method=method)





# alignment methods related to convolution -- need work!
#
def alignment_convolutional(activity, weight, each_look=True):
    h_max, w_max = get_maximum_strides(activity.shape[2], activity.shape[3], weight)
    
# # similarity / alignment for convolutional network layers
# def alignmentConvLayer(inputActivity, layer, eachLook=True):
#     hMax, wMax = getMaximumStrides(inputActivity.shape[2], inputActivity.shape[3], layer)
#     if eachLook:
#         preprocess0 = transforms.Pad(layer.padding)
#         pInput0 = preprocess0(inputActivity).detach()
#         numLooks = hMax * wMax
#         numChannels = layer.out_channels
#         alignLayer = torch.empty((numChannels, numLooks))
#         for h in range(hMax): 
#             for w in range(wMax):
#                 alignLayer[:,w+hMax*h] = alignmentConvLook(pInput0, layer, (h,w))
#         return alignLayer
#     else:
#         unfoldedInput = F.unfold(inputActivity.detach(), layer.kernel_size, stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
#         unfoldedInput = unfoldedInput.transpose(1,2).reshape(inputActivity.size(0), -1)
#         unfoldedWeight = layer.weight.data.detach().view(layer.weight.size(0), -1).repeat(1, hMax*wMax)
#         return alignment(unfoldedInput, unfoldedWeight)

# def alignmentConvLook(pInput, layer, stride):
#     # Take (NI, C, H, W) (transformed) input activity
#     # And compute the similarity for one "look" e.g. one position of the convolutional filter
#     numImages = pInput.shape[0]
#     wIdxStart, hIdxStart = np.meshgrid(np.arange(0,layer.kernel_size[0]), np.arange(0, layer.kernel_size[1]))
#     numElements = pInput.shape[1] * layer.kernel_size[0] * layer.kernel_size[1]
#     alignedInput = pInput[:,:,hIdxStart + stride[0]*layer.stride[0], wIdxStart + stride[1]*layer.stride[1]].reshape(numImages, numElements)
#     alignedWeights = layer.weight.data.reshape(layer.out_channels, numElements).detach()
#     return alignment(alignedInput, alignedWeights, method='alignment')

# def similarityConvLayer(inputActivity, layer):
#     preprocess0 = transforms.Pad(layer.padding)
#     pInput0 = preprocess0(inputActivity).detach()
#     hMax, wMax = get_maximum_strides(inputActivity.shape[2], inputActivity.shape[3], layer)
#     numLooks = hMax * wMax
#     numChannels = layer.out_channels
#     simLayer = torch.empty((numChannels, numLooks))
#     for h in range(hMax): 
#         for w in range(wMax):
#             simLayer[:,w+hMax*h] = similarityConvLook(pInput0, layer, (h,w))
#     return simLayer

# def similarityConvLook(pInput, layer, stride):
#     # Take (NI, C, H, W) (transformed) input activity
#     # And compute the similarity for one "look" e.g. one position of the convolutional filter
#     numImages = pInput.shape[0]
#     wIdxStart, hIdxStart = np.meshgrid(np.arange(0,layer.kernel_size[0]), np.arange(0, layer.kernel_size[1]))
#     numElements = pInput.shape[1] * layer.kernel_size[0] * layer.kernel_size[1]
#     alignedInput = pInput[:,:,hIdxStart + stride[0]*layer.stride[0], wIdxStart + stride[1]*layer.stride[1]].reshape(numImages, numElements)
#     alignedWeights = layer.weight.data.reshape(layer.out_channels, numElements).detach()
#     return alignment(alignedInput, alignedWeights, method='similarity')

def get_maximum_strides(hInput, wInput, layer):
    hMax = int(np.floor((hInput + 2*layer.padding[0] - layer.dilation[0]*(layer.kernel_size[0] - 1) -1)/layer.stride[0] + 1))
    wMax = int(np.floor((wInput + 2*layer.padding[1] - layer.dilation[1]*(layer.kernel_size[1] - 1) -1)/layer.stride[1] + 1))
    return hMax, wMax
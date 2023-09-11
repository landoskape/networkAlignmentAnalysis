import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
    
def alignment(inputActivity, weights, method='alignment'):
    # input activity is a (b x n) matrix, where b=batch and n=neurons
    # weights is a (m x n) matrix where each row corresponds to the weights for a single postsynaptic neuron
    # computes the rayleigh quotient R(cc, w) between the weight of each postsynaptic neuron and the correlation matrix of the inputs 
    # then divides by n, because sum(eigenvalue)=sum(trace)=n, so this bounds the outputs between 0 and 1
    # -- note: this breaks if an input element has a standard deviation of 0! so we're just ignoring those values --
    assert method=='alignment' or method=='similarity', "method must be set to either 'alignment' or 'similarity' (or None, default is alignment)"
    inputActivity = torch.tensor(inputActivity) if not torch.is_tensor(inputActivity) else inputActivity
    weights = torch.tensor(weights) if not torch.is_tensor(weights) else weights
    b,n = inputActivity.shape
    m = weights.shape[1]
    if method=='alignment':
        cc = torch.cov(inputActivity.T)
    elif method=='similarity':
        idxMute = torch.where(torch.std(inputActivity,axis=0)==0)[0]
        cc = torch.corrcoef(inputActivity.T)
        cc[idxMute,:] = 0
        cc[:,idxMute] = 0
    else: 
        raise ValueError("did not recognize method")
    # Compute rayleigh quotient
    rq = torch.sum(torch.matmul(weights, cc) * weights, axis=1) / torch.sum(weights * weights, axis=1)
    # proportion of variance explained by a projection of the input onto each weight
    return rq/torch.trace(cc)

# similarity / alignment for linear network layers
def similarityLinearLayer(inputActivity, layer):
    return alignment(inputActivity.detach(), layer.weight.data.detach(), method='similarity')

def alignmentLinearLayer(inputActivity, layer):
    return alignment(inputActivity.detach(), layer.weight.data.detach(), method='alignment')

# similarity / alignment for convolutional network layers
def alignmentConvLayer(inputActivity, layer, eachLook=True):
    hMax, wMax = getMaximumStrides(inputActivity.shape[2], inputActivity.shape[3], layer)
    if eachLook:
        preprocess0 = transforms.Pad(layer.padding)
        pInput0 = preprocess0(inputActivity).detach()
        numLooks = hMax * wMax
        numChannels = layer.out_channels
        alignLayer = torch.empty((numChannels, numLooks))
        for h in range(hMax): 
            for w in range(wMax):
                alignLayer[:,w+hMax*h] = alignmentConvLook(pInput0, layer, (h,w))
        return alignLayer
    else:
        unfoldedInput = F.unfold(inputActivity.detach(), layer.kernel_size, stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
        unfoldedInput = unfoldedInput.transpose(1,2).reshape(inputActivity.size(0), -1)
        unfoldedWeight = layer.weight.data.detach().view(layer.weight.size(0), -1).repeat(1, hMax*wMax)
        return alignment(unfoldedInput, unfoldedWeight)

def alignmentConvLook(pInput, layer, stride):
    # Take (NI, C, H, W) (transformed) input activity
    # And compute the similarity for one "look" e.g. one position of the convolutional filter
    numImages = pInput.shape[0]
    wIdxStart, hIdxStart = np.meshgrid(np.arange(0,layer.kernel_size[0]), np.arange(0, layer.kernel_size[1]))
    numElements = pInput.shape[1] * layer.kernel_size[0] * layer.kernel_size[1]
    alignedInput = pInput[:,:,hIdxStart + stride[0]*layer.stride[0], wIdxStart + stride[1]*layer.stride[1]].reshape(numImages, numElements)
    alignedWeights = layer.weight.data.reshape(layer.out_channels, numElements).detach()
    return alignment(alignedInput, alignedWeights, method='alignment')

def similarityConvLayer(inputActivity, layer):
    preprocess0 = transforms.Pad(layer.padding)
    pInput0 = preprocess0(inputActivity).detach()
    hMax, wMax = getMaximumStrides(inputActivity.shape[2], inputActivity.shape[3], layer)
    numLooks = hMax * wMax
    numChannels = layer.out_channels
    simLayer = torch.empty((numChannels, numLooks))
    for h in range(hMax): 
        for w in range(wMax):
            simLayer[:,w+hMax*h] = similarityConvLook(pInput0, layer, (h,w))
    return simLayer

def similarityConvLook(pInput, layer, stride):
    # Take (NI, C, H, W) (transformed) input activity
    # And compute the similarity for one "look" e.g. one position of the convolutional filter
    numImages = pInput.shape[0]
    wIdxStart, hIdxStart = np.meshgrid(np.arange(0,layer.kernel_size[0]), np.arange(0, layer.kernel_size[1]))
    numElements = pInput.shape[1] * layer.kernel_size[0] * layer.kernel_size[1]
    alignedInput = pInput[:,:,hIdxStart + stride[0]*layer.stride[0], wIdxStart + stride[1]*layer.stride[1]].reshape(numImages, numElements)
    alignedWeights = layer.weight.data.reshape(layer.out_channels, numElements).detach()
    return similarity(alignedInput, alignedWeights, method='similarity')

def getMaximumStrides(hInput, wInput, layer):
    hMax = int(np.floor((hInput + 2*layer.padding[0] - layer.dilation[0]*(layer.kernel_size[0] - 1) -1)/layer.stride[0] + 1))
    wMax = int(np.floor((wInput + 2*layer.padding[1] - layer.dilation[1]*(layer.kernel_size[1] - 1) -1)/layer.stride[1] + 1))
    return hMax, wMax
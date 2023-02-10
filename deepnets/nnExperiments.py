import numpy as np
import os
import time
import tqdm
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import deepnets.nnModels as models

def runExperiment_measureIntegrationMNIST(useNet='CNN32',DEVICE=None,iterations=10,learningRate=1e-2,verbose=True):
    """
    Function that measures integration across training of a neural network in each layer while learning to classify MNIST
    Input: 
    Output:
    """
    
    # First, select device to run networks on
    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU = "cpu"
    print(f"Initiating experiment using {DEVICE}")
    
    # Prepare datasets
    batchSize = 1000
    preprocess = transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
    ])
    trainset, testset, trainloader, testloader, numClasses = downloadMNIST(batchSize=batchSize, preprocess=preprocess)
    
    # Prepare Network
    weightvars = None # initialize variance of weights with default parameters
    convActivation = F.relu
    linearActivation = F.relu
    if useNet=='CNN32':
        net = models.CNN32(weightvars=weightvars,convActivation=convActivation,linearActivation=linearActivation)
    elif useNet=='CNN2P2':
        net = models.CNN2P2(weightvars=weightvars,convActivation=convActivation,linearActivation=linearActivation)
    else:
        raise ValueError('useNet not recognized')
    net.to(DEVICE)
    
    # Prepare Training Functions 
    loss_function = nn.CrossEntropyLoss() # Note: this automatically applies softmax...
    optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)
    # optimizer = torch.optim.Adadelta(net.parameters())
    
    # Preallocate summary variables  
    numTrainingSteps = len(trainloader)*iterations
    trackLoss = torch.zeros(numTrainingSteps)
    trackAccuracy = torch.zeros(numTrainingSteps)
    alignFull = []
    deltaWeights = []
    
    initWeights = net.getNetworkWeights()
    
    # Train Network & Measure Integration
    t = time.time()
    for epoch in range(0, iterations): 
        # Set current loss value
        currentLoss = 0.0
        numBatches = 0
        currentCorrect = 0
        currentAttempted = 0

        for idx,batch in enumerate(trainloader):
            cidx = epoch*len(trainloader) + idx
            
            images, label = batch
            images = images.to(DEVICE)
            label = label.to(DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = net(images)

            # Perform backward pass & optimization
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
            
            # Measure Integration
            alignFull.append(net.measureAlignment(images))
            
            # Measure Change in Weights
            deltaWeights.append(net.compareNetworkWeights(initWeights))

            # Track Loss and Accuracy
            trackLoss[cidx] = loss.item()
            trackAccuracy[cidx] = 100*torch.sum(torch.argmax(outputs,axis=1)==label)/images.shape[0]

        # Print statistics for each epoch
        if verbose: print('Loss in epoch %3d: %.3f, Accuracy: %.2f%%.' % (epoch, loss.item(), 100*torch.sum(torch.argmax(outputs,axis=1)==label)/images.shape[0]))

    # Measure performance on test set
    totalLoss = 0
    numCorrect = 0
    numAttempted = 0
    for batch in testloader:
        images, label = batch
        images = images.to(DEVICE)
        label = label.to(DEVICE)
        outputs = net(images)
        totalLoss += loss_function(outputs,label).item()
        output1 = torch.argmax(outputs,axis=1)
        numCorrect += sum(output1==label)
        numAttempted += images.shape[0]

    print(f"Average loss over test set: %.2f." % (totalLoss / len(testloader)))
    print(f"Accuracy over test set: %.2f%%." % (100*numCorrect/numAttempted))
    print('Training process has finished in %.3f seconds.' % (time.time()-t))
    
    results = {
        'net':net,
        'initWeights':initWeights,
        'alignFull':alignFull,
        'deltaWeights':deltaWeights,
        'trackLoss':trackLoss,
        'trackAccuracy':trackAccuracy,
        'trainset':trainset,
        'testset':testset,
        'trainloader':trainloader,
        'testloader':testloader,
        'learningRate':learningRate,
    }
    return results


def downloadMNIST(batchSize=1000,preprocess=None):
    dataPath = getDataPath()
    trainset = torchvision.datasets.MNIST(root=dataPath, train=True, download=True, transform=preprocess)
    testset = torchvision.datasets.MNIST(root=dataPath, train=False, download=True, transform=preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True, num_workers=2)
    numClasses = 10
    return trainset, testset, trainloader, testloader, numClasses
    
def getDataPath():
    # Path to stored datasets (might add input argument for running on a different computer...)
    return os.path.join('C:/', 'Users','andrew','Documents','machineLearning','datasets')


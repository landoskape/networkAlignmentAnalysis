import numpy as np
import os
import time
from tqdm import tqdm
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import deepnets.nnModels as models

def runExperiment_measureIntegrationMNIST(useNet='CNN32',DEVICE=None,iterations=10,learningRate=1e-2,verbose=True,dataPath=None):
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
    trainset, testset, trainloader, testloader, numClasses = downloadMNIST(batchSize=batchSize, preprocess=preprocess, dataPath=dataPath)
    
    # Prepare Network
    weightvars = None # initialize variance of weights with default parameters
    convActivation = F.relu
    linearActivation = F.relu
    if useNet=='CNN2P2':
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

def retrainNetwork(trainedNet, dropoutLayer, dropoutIndex, trainloader, testloader, DEVICE=None, iterations=10, learningRate=1e-2, verbose=True):
    # takes a net that has been trained, along with instructions for dropout (dropoutLayer & dropoutIndex)
    # dropoutLayer is a tuple of integers indicating which layer to do the dropout for
    # dropoutIndex is a tuple of index arrays indicating which nodes to dropout in the associated layer
    # -
    # this function constructs a new network, assigns the appropriate learned weights and biases to it (not including the ones that are dropped), and retrains as before
    
    # Check Input
    assert len(dropoutLayer)==len(dropoutIndex), "Dropout layer and dropout index must be the same size"
    
    # First, select device to run networks on
    if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU = "cpu"
    print(f"Initiating retraining experiment using {DEVICE}")
    
    # Create new network
    net = models.CNN2P2(weightvars=None,convActivation=trainedNet.convActivation,linearActivation=trainedNet.linearActivation)
    net.to(DEVICE)
    
    # Copy data from trained net to new network
    net.conv1.weight.data = trainedNet.conv1.weight.data.clone().detach()
    net.conv1.bias.data = trainedNet.conv1.bias.data.clone().detach()
    net.conv2.weight.data = trainedNet.conv2.weight.data.clone().detach()
    net.conv2.bias.data = trainedNet.conv2.bias.data.clone().detach()
    net.fc1.weight.data = trainedNet.fc1.weight.data.clone().detach()
    net.fc1.bias.data = trainedNet.fc1.bias.data.clone().detach()
    net.o.weight.data = trainedNet.o.weight.data.clone().detach()
    net.o.bias.data = trainedNet.o.bias.data.clone().detach()
    
    # Prune network 
    for doLayer,doIndex in zip(dropoutLayer, dropoutIndex):
        if doLayer==0:
            idxChannels = np.arange(len(net.conv1.bias.data))
            idxKeep = np.where(~np.isin(idxChannels, doIndex))[0]
            net.conv1.weight.data = net.conv1.weight.data[idxKeep]
            net.conv1.bias.data = net.conv1.bias.data[idxKeep]
            net.conv2.weight.data = net.conv2.weight.data[:,idxKeep]
            net.conv1.out_channels = len(idxKeep)
            net.conv2.in_channels = len(idxKeep)
            
        elif doLayer==1:
            idxChannels = np.arange(len(net.conv2.bias.data))
            idxKeep = np.where(~np.isin(idxChannels, doIndex))[0]
            net.conv2.weight.data = net.conv2.weight.data[idxKeep]
            net.conv2.bias.data = net.conv2.bias.data[idxKeep]
            net.conv2.out_channels = len(idxKeep)
            
            # And also update input to following layer
            outputPerChannel = 4 # fixed based on shape of MNIST images, first two convolutional layers, and maxpooling layer
            idxInputsFC = np.arange(net.fc1.weight.shape[1])
            idxRemove = []
            for doidx in doIndex:
                idxRemove.append(np.arange(outputPerChannel) + outputPerChannel*np.array(doidx))
            idxRemove = np.array(idxRemove)
            idxKeep = np.where(~np.isin(idxInputsFC, idxRemove))[0]
            net.fc1.weight.data = net.fc1.weight.data[:,idxKeep]
            
            net.fc1.in_channels = len(idxKeep)*outputPerChannel
            
        elif doLayer==2:
            idxChannels = np.arange(len(net.fc1.bias.data))
            idxKeep = np.where(~np.isin(idxChannels, doIndex))[0]
            net.fc1.weight.data = net.fc1.weight.data[idxKeep]
            net.fc1.bias.data = net.fc1.bias.data[idxKeep]
            net.o.weight.data = net.o.weight.data[:,idxKeep]
            net.fc1.out_channels = len(idxKeep)
            net.o.in_channels = len(idxKeep)
            
        else:
            raise ValueError("Can only perform dropout in layers 0, 1, & 2!")
    
    postLoss, postAccuracy = measurePerformance(net, testloader, verbose=False)
    print(f"Post prune loss:{postLoss:.2f}, Post prune accuracy:{postAccuracy:.1f}%")

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
    print('Retraining process has finished in %.3f seconds.' % (time.time()-t))
    
    results = {
        'net':net,
        'initWeights':initWeights,
        'alignFull':alignFull,
        'deltaWeights':deltaWeights,
        'trackLoss':trackLoss,
        'trackAccuracy':trackAccuracy,
        'trainloader':trainloader,
        'testloader':testloader,
        'learningRate':learningRate,
    }
    return results

def measurePerformance(net, dataloader, DEVICE=None, verbose=False):
    if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Measure performance
    loss_function = nn.CrossEntropyLoss()
    totalLoss = 0
    numCorrect = 0
    numAttempted = 0
    
    if verbose: iterator = tqdm(dataloader)
    else: iterator = dataloader
    
    for batch in iterator:
        images, label = batch
        images = images.to(DEVICE)
        label = label.to(DEVICE)
        outputs = net(images)
        totalLoss += loss_function(outputs,label).item()
        output1 = torch.argmax(outputs,axis=1)
        numCorrect += sum(output1==label)
        numAttempted += images.shape[0]
        
    return totalLoss/len(dataloader), 100*numCorrect/numAttempted

def downloadMNIST(batchSize=1000,preprocess=None,dataPath=None):
    dataPath = getDataPath(dataPath)
    trainset = torchvision.datasets.MNIST(root=dataPath, train=True, download=True, transform=preprocess)
    testset = torchvision.datasets.MNIST(root=dataPath, train=False, download=True, transform=preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True, num_workers=2)
    numClasses = 10
    return trainset, testset, trainloader, testloader, numClasses
    
def getDataPath(dataPath=None):
    # Path to stored datasets (might add input argument for running on a different computer...)
    if dataPath is None: 
        return os.path.join('C:/', 'Users','andrew','Documents','machineLearning','datasets')
    if dataPath=="colab":
        return "/content/drive/MyDrive/machineLearningDatasets"


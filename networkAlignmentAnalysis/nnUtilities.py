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

def trainNetwork(net, dataloader, lossFunction, optimizer, iterations, DEVICE, verbose=False):
    """
    Generic function for training network and measuring alignment throughout 
    """
    
    # Preallocate summary variables  
    numTrainingSteps = len(dataloader)*iterations
    trackLoss = torch.zeros(numTrainingSteps)
    trackAccuracy = torch.zeros(numTrainingSteps)
    allWeights = []
    alignFull = []
    
    initWeights = net.getNetworkWeights()
    
    # Train Network & Measure Integration
    t = time.time()
    for epoch in range(0, iterations): 
        # Set current loss value
        currentLoss = 0.0
        numBatches = 0
        currentCorrect = 0
        currentAttempted = 0

        for idx,batch in enumerate(dataloader):
            cidx = epoch*len(dataloader) + idx
            
            images, label = batch
            images = images.to(DEVICE)
            label = label.to(DEVICE)

            # Zero the gradients
            net.zero_grad()
            optimizer.zero_grad()

            # Perform forward pass
            outputs = net(images)
            
            # Perform backward pass & optimization
            loss = lossFunction(outputs, label)
            loss.backward()
            optimizer.step()

            # Measure Integration
            alignFull.append(net.measureAlignment(images))
            
            # Track Loss and Accuracy
            trackLoss[cidx] = loss.item()
            trackAccuracy[cidx] = 100*torch.sum(torch.argmax(outputs,axis=1)==label)/images.shape[0]
        
        # Return current weights
        allWeights.append([cw.cpu() for cw in net.getNetworkWeights()])
            
        # Print statistics for each epoch
        if verbose: print('Loss in epoch %3d: %.3f, Accuracy: %.2f%%.' % (epoch, loss.item(), 100*torch.sum(torch.argmax(outputs,axis=1)==label)/images.shape[0]))
    
    results = {
        'net':net,
        'initWeights':initWeights,
        'allWeights':allWeights,
        'trackLoss':trackLoss,
        'trackAccuracy':trackAccuracy,
        'alignFull':alignFull,
    }
    return results

def trainNetworkRichInfo(net, dataloader, lossFunction, optimizer, iterations, DEVICE, verbose=False):
    """
    Generic function for training network and measuring alignment throughout 
    """
    
    # Preallocate summary variables  
    numTrainingSteps = len(dataloader)*iterations
    trackLoss = torch.zeros(numTrainingSteps)
    trackAccuracy = torch.zeros(numTrainingSteps)
    allWeights = []
    alignFull = []
    deltaWeights = []
    betas = []
    evals = []
    evecs = []
    
    initWeights = net.getNetworkWeights()
    
    # Train Network & Measure Integration
    t = time.time()
    for epoch in range(0, iterations): 
        # Set current loss value
        currentLoss = 0.0
        numBatches = 0
        currentCorrect = 0
        currentAttempted = 0

        for idx,batch in enumerate(dataloader):
            cidx = epoch*len(dataloader) + idx
            
            images, label = batch
            images = images.to(DEVICE)
            label = label.to(DEVICE)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = net(images)
            
            # Perform backward pass & optimization
            loss = lossFunction(outputs, label)
            loss.backward()
            optimizer.step()
            
            # Measure Integration
            alignFull.append(net.measureAlignment(images))
            
            # Measure Change in Weights (NORM)
            deltaWeights.append(net.compareNetworkWeights(initWeights))

            # Track Loss and Accuracy
            trackLoss[cidx] = loss.item()
            trackAccuracy[cidx] = 100*torch.sum(torch.argmax(outputs,axis=1)==label)/images.shape[0]

        # Print statistics for each epoch
        if verbose: print('Loss in epoch %3d: %.3f, Accuracy: %.2f%%.' % (epoch, loss.item(), 100*torch.sum(torch.argmax(outputs,axis=1)==label)/images.shape[0]))
        
        # Return current weights (too much data if we do this every time)
        allWeights.append([cw.cpu() for cw in net.getNetworkWeights()])
            
        # Measure eigenfeatures after each round through the data
        cbetas, cevals, cevecs = net.measureEigenFeatures(net, dataloader)
        betas.append([cb.cpu() for cb in cbetas])
        evals.append(cevals)
        evecs.append(cevecs)
    
    results = {
        'net':net,
        'initWeights':initWeights,
        'allWeights':allWeights,
        'trackLoss':trackLoss,
        'trackAccuracy':trackAccuracy,
        'alignFull':alignFull,
        'deltaWeights':deltaWeights,
        'beta':betas,
        'evals':evals,
        'evecs':evecs,
    }
    return results

def trainNetworkManualShape(net, dataloader, lossFunction, optimizer, iterations, DEVICE, verbose=False, doManual=True, evalTransform=None):
    """
    Generic function for training network and measuring alignment throughout 
    """
    
    # Preallocate summary variables  
    numTrainingSteps = len(dataloader)*iterations
    trackLoss = torch.zeros(numTrainingSteps)
    trackAccuracy = torch.zeros(numTrainingSteps)
    allWeights = []
    alignFull = []
    betas = []
    evals = []
    evecs = []
    
    initWeights = net.getNetworkWeights()
    
    # Train Network & Measure Integration
    t = time.time()
    for epoch in range(0, iterations): 
        # Set current loss value
        currentLoss = 0.0
        numBatches = 0
        currentCorrect = 0
        currentAttempted = 0

        for idx,batch in enumerate(dataloader):
            cidx = epoch*len(dataloader) + idx
            
            images, label = batch
            images = images.to(DEVICE)
            label = label.to(DEVICE)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = net(images)
            
            # Perform backward pass & optimization
            loss = lossFunction(outputs, label)
            loss.backward()
            optimizer.step()
            
            # Measure Integration
            alignFull.append(net.measureAlignment(images))

            # Track Loss and Accuracy
            trackLoss[cidx] = loss.item()
            trackAccuracy[cidx] = 100*torch.sum(torch.argmax(outputs,axis=1)==label)/images.shape[0]

        # Print statistics for each epoch
        if verbose: print('Loss in epoch %3d: %.3f, Accuracy: %.2f%%.' % (epoch, loss.item(), 100*torch.sum(torch.argmax(outputs,axis=1)==label)/images.shape[0]))
            
        # Measure eigenfeatures after each round through the data
        cbetas, cevals, cevecs = net.measureEigenFeatures(net, dataloader)
        betas.append([cb.cpu() for cb in cbetas])
        evals.append(cevals)
        evecs.append(cevecs)
        
        # Implement manual shape
        if doManual:
            net.manualShape(cevals, cevecs, DEVICE, evalTransform=evalTransform)
            
        # Return current weights (too much data if we do this every time)
        allWeights.append([cw.cpu() for cw in net.getNetworkWeights()])
            
    
    results = {
        'net':net,
        'initWeights':initWeights,
        'trackLoss':trackLoss,
        'trackAccuracy':trackAccuracy,
        'alignFull':alignFull,
        'allWeights':allWeights,
        'beta':betas,
        'evals':evals,
        'evecs':evecs,
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

def downloadMNIST(batchSize=1000,preprocess=None):
    dataPath = getDataPath('MNIST')
    trainset = torchvision.datasets.MNIST(root=dataPath, train=True, download=True, transform=preprocess)
    testset = torchvision.datasets.MNIST(root=dataPath, train=False, download=True, transform=preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True, num_workers=2)
    numClasses = 10
    return trainloader, testloader, numClasses

def downloadImageNet(batchSize=1000):
    dataPath = getDataPath('ImageNet')
    valTransform = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
    # valTransform = transforms.Compose([
    #     transforms.Resize(256,interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    #     ])
    valData = torchvision.datasets.ImageNet(dataPath,split='val',transform=valTransform)
    valLoader = torch.utils.data.DataLoader(valData, batch_size=500, shuffle=True, num_workers=2, pin_memory=False)
    return valLoader

def getDataPath(dataset='MNIST'):
    # Path to stored datasets (might add input argument for running on a different computer...)
    if dataset=='MNIST':
        return os.path.join('C:/', 'Users','andrew','Documents','machineLearning','datasets')
    elif dataset=='ImageNet':
        return os.path.join('C:/', 'Users','andrew','Documents','machineLearning','datasets','imagenet')
    else: 
        raise ValueError("Didn't recognize dataset string.")
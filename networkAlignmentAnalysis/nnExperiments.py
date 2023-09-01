import numpy as np
import os
import time
from tqdm import tqdm
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import deepnets.nnUtilities as nnutils
import deepnets.nnModels as models
import deepnets.nnEigenModels as eigModels

def runExperiment_measureIntegrationMNIST(useNet='CNN32',DEVICE=None,iterations=10,learningRate=1e-2,verbose=True,doInitWeights=False,ojaAlpha=0,pDropout=0.5):
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
    
    # Prepare Network and Preprocessing
    batchSize = 1000
    weightvars = None # initialize variance of weights with default parameters
    if useNet=='CNN2P2':
        convActivation = F.relu
        linearActivation = F.relu
        net = models.CNN2P2(convActivation=convActivation,linearActivation=linearActivation)
        
        preprocess = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
        ])
    elif useNet=='MLP3':
        actFunc = F.relu
        net = models.MLP3(actFunc=actFunc)
        
        preprocess = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
            transforms.Lambda(torch.flatten), # convert to vectors
        ])
    elif useNet=='MLP4':
        actFunc = F.relu
        net = models.MLP4(actFunc=actFunc,pDropout=pDropout)
        
        preprocess = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
            transforms.Lambda(torch.flatten), # convert to vectors
        ])
    elif useNet=='eigNet':
        actFunc = F.relu
        preprocess = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
            transforms.Lambda(torch.flatten), # convert to vectors
        ])
    else:
        raise ValueError('useNet not recognized')
    
    # Prepare Dataloaders
    trainloader, testloader, numClasses = nnutils.downloadMNIST(batchSize=batchSize, preprocess=preprocess)
    
    if useNet=='eigNet':
        # Create Network
        eigdata = torch.cat([batch[0].to(DEVICE) for batch in trainloader])
        net = eigModels.eigenNet(eigdata,actFunc=actFunc) # use testloader for fitting eigenstructure to speed it up...
        net.zero_grad()

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
            
            if ojaAlpha>0:
                net.doOjaUpdate(images, learningRate*ojaAlpha)
            
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
        'trainloader':trainloader,
        'testloader':testloader,
        'learningRate':learningRate,
    }
    return results

def retrainNetwork(trainedNet, useNet, dropoutLayer, dropoutIndex, trainloader, testloader, DEVICE=None, iterations=10, learningRate=1e-2, verbose=True):
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
    if useNet=='CNN2P2':
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
                
    elif useNet=='MLP4':
        net = models.MLP4(weightvars=None,actFunc=trainedNet.actFunc)
        net.to(DEVICE)
        
        # Copy data from trained net to new network
        net.fc1.weight.data = trainedNet.fc1.weight.data.clone().detach()
        net.fc1.bias.data = trainedNet.fc1.bias.data.clone().detach()
        net.fc2.weight.data = trainedNet.fc2.weight.data.clone().detach()
        net.fc2.bias.data = trainedNet.fc2.bias.data.clone().detach()
        net.fc3.weight.data = trainedNet.fc3.weight.data.clone().detach()
        net.fc3.bias.data = trainedNet.fc3.bias.data.clone().detach()
        net.fc4.weight.data = trainedNet.fc4.weight.data.clone().detach()
        net.fc4.bias.data = trainedNet.fc4.bias.data.clone().detach()

        # Prune network 
        for doLayer,doIndex in zip(dropoutLayer, dropoutIndex):
            if doLayer==0:
                idxChannels = np.arange(len(net.fc1.bias.data))
                idxKeep = np.where(~np.isin(idxChannels, doIndex))[0]
                net.fc1.weight.data = net.fc1.weight.data[idxKeep]
                net.fc1.bias.data = net.fc1.bias.data[idxKeep]
                net.fc2.weight.data = net.fc2.weight.data[:,idxKeep]
                net.fc1.out_channels = len(idxKeep)
                net.fc2.in_channels = len(idxKeep)

            elif doLayer==1:
                idxChannels = np.arange(len(net.fc2.bias.data))
                idxKeep = np.where(~np.isin(idxChannels, doIndex))[0]
                net.fc2.weight.data = net.fc2.weight.data[idxKeep]
                net.fc2.bias.data = net.fc2.bias.data[idxKeep]
                net.fc3.weight.data = net.fc3.weight.data[:,idxKeep]
                net.fc2.out_channels = len(idxKeep)
                net.fc3.in_channels = len(idxKeep)

            elif doLayer==2:
                idxChannels = np.arange(len(net.fc3.bias.data))
                idxKeep = np.where(~np.isin(idxChannels, doIndex))[0]
                net.fc3.weight.data = net.fc3.weight.data[idxKeep]
                net.fc3.bias.data = net.fc3.bias.data[idxKeep]
                net.fc4.weight.data = net.fc4.weight.data[:,idxKeep]
                net.fc3.out_channels = len(idxKeep)
                net.fc4.in_channels = len(idxKeep)

            else:
                raise ValueError("Can only perform dropout in layers 0, 1, & 2!")
    
    postLoss, postAccuracy = nnutils.measurePerformance(net, testloader, verbose=False)
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


# Experiment to compare alignment with and without dropout. 
# Trains MLP or CNN on the MNIST task, measures alignment throughout training, then turns on dropout and does the same thing.  
def compareAlignmentDropout(useNet='MLP4',useNetID=None,DEVICE=None,iterations=10,learningRate=1e-2,pDropout=0.5,weightDecay=1e-3,verbose=True,richInfo=False):
    """
    # Experiment to compare alignment with and without dropout. 
    # Trains MLP or CNN or eigNet on the MNIST task, measures alignment throughout training, then turns on dropout and does the same thing.      
    """
    
    assert len(pDropout)==len(weightDecay), "pDropout and weightDecay have to be the same length"
    
    # First, select device to run networks on
    if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU = "cpu"
    print(f"Initiating experiment using {DEVICE}")
    
    # Prepare Network and Preprocessing
    batchSize = 500
    
    # Prepare Training Functions 
    lossFunction = nn.CrossEntropyLoss() # Note: this automatically applies softmax...
    
    fullResults = []
    
    # Train Networks without dropout
    numRuns = len(pDropout)
    for runidx, prms in enumerate(zip(pDropout, weightDecay, useNetID)):
        cDropout, cWeightDecay, cUseNetID = prms
        
        t = time.time()
        
        # Create new network
        if useNet[cUseNetID]=='CNN2P2':
            convActivation = F.relu
            linearActivation = F.relu        
            preprocess = transforms.Compose([
                transforms.ToTensor(), # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
            ])
            net = models.CNN2P2(convActivation=convActivation,linearActivation=linearActivation)
            net.to(DEVICE)
            # Prepare Dataloaders
            trainloader, testloader, numClasses = nnutils.downloadMNIST(batchSize=batchSize, preprocess=preprocess)
        
        elif useNet[cUseNetID]=='MLP4':
            actFunc = F.relu        
            preprocess = transforms.Compose([
                transforms.ToTensor(), # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
                transforms.Lambda(torch.flatten), # convert to vectors
            ])
            net = models.MLP4(actFunc=actFunc,pDropout=cDropout)
            net.to(DEVICE)
            # Prepare Dataloaders
            trainloader, testloader, numClasses = nnutils.downloadMNIST(batchSize=batchSize, preprocess=preprocess)
        
        elif useNet[cUseNetID]=='eigNet':
            actFunc = F.relu        
            preprocess = transforms.Compose([
                transforms.ToTensor(), # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
                transforms.Lambda(torch.flatten), # convert to vectors
            ])
            # Prepare Dataloaders
            trainloader, testloader, numClasses = nnutils.downloadMNIST(batchSize=batchSize, preprocess=preprocess)
            eigdata = torch.cat([batch[0].to(DEVICE) for batch in trainloader])
            net = eigModels.eigenNet(eigdata,actFunc=actFunc,pDropout=cDropout) # use testloader for fitting eigenstructure to speed it up...
        else: 
            raise ValueError("useNet not recognized")
        
        # Set up optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, weight_decay=cWeightDecay)
        # optimizer = torch.optim.Adadelta(net.parameters(), weight_decay=cWeightDecay)
        
        # Train network (richInfo switch measures the eigenfeatures at every epoch of training)
        if richInfo:
            results = nnutils.trainNetworkRichInfo(net, trainloader, lossFunction, optimizer, iterations, DEVICE=DEVICE, verbose=verbose)
        else:
            results = nnutils.trainNetwork(net, trainloader, lossFunction, optimizer, iterations, DEVICE=DEVICE, verbose=verbose)
        
        # remove dropout and then measuring test performance
        storeDropout = results['net'].getDropout()
        results['net'].setDropout(0)
        results['trainloader'] = trainloader
        results['testloader'] = testloader
        results['pDropout'] = cDropout
        results['cWeightDecay'] = cWeightDecay
        testLoss, testAccuracy = nnutils.measurePerformance(results['net'], testloader, DEVICE=DEVICE, verbose=False)
        print(f"Network {runidx+1}/{numRuns} finished in {time.time()-t:.1f} seconds. pDropout:{cDropout:.1f}. WeightDecay:{cWeightDecay}. Test loss:{testLoss:.2f}. Test accuracy: {testAccuracy:.2f}%.")
        results['net'].setDropout(storeDropout)
        
        # Save Everything, delete unnecessary, continue
        fullResults.append(results)
        del net, results
    
    return fullResults

# Experiment to measure effect of shaping the alignment on network performance and training speed. 
# Will do so in two ways: 
#    1. Manually remove eigenvectors in weight matrices according to some variance schedule
#    2. Normalize weights? (I have a hunch that this will focus the learning on the significant dimensions)
def measureAlignmentShaping(useNet='MLP4',DEVICE=None,iterations=10,learningRate=1e-2,learningParameters={},verbose=True,richInfo=False):
    """
    # Experiment to compare alignment with and without dropout. 
    # Trains MLP or CNN on the MNIST task, measures alignment throughout training, then turns on dropout and does the same thing.      
    """
    
    
    # First, select device to run networks on
    if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU = "cpu"
    print(f"Initiating experiment using {DEVICE}")
    
    # Prepare Network and Preprocessing
    batchSize = 500
    if useNet=='CNN2P2':
        convActivation = F.relu
        linearActivation = F.relu        
        preprocess = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
        ])
    elif useNet=='MLP4':
        actFunc = F.relu        
        preprocess = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
            transforms.Lambda(torch.flatten), # convert to vectors
        ])
    else:
        raise ValueError('useNet not recognized')
    
    # Prepare Dataloaders
    trainloader, testloader, numClasses = nnutils.downloadMNIST(batchSize=batchSize, preprocess=preprocess)
        
    # Prepare Training Functions 
    lossFunction = nn.CrossEntropyLoss() # Note: this automatically applies softmax...
    
    fullResults = []
    
    # Train Networks without dropout
    numRuns = len(pDropout)
    for runidx, prms in enumerate(zip(pDropout, weightDecay)):
        cDropout, cWeightDecay = prms
        
        t = time.time()
        
        # Create new network
        if useNet=='CNN2P2':
            net = models.CNN2P2(convActivation=convActivation,linearActivation=linearActivation)
            net.to(DEVICE)
        elif useNet=='MLP4':
            net = models.MLP4(actFunc=actFunc,pDropout=cDropout)
            net.to(DEVICE)
        else: 
            raise ValueError("useNet not recognized")
        
        # Set up optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, weight_decay=cWeightDecay)
        # optimizer = torch.optim.Adadelta(net.parameters(), weight_decay=cWeightDecay)
        
        # Train network (richInfo switch measures the eigenfeatures at every epoch of training)
        if richInfo:
            results = nnutils.trainNetworkRichInfo(net, trainloader, lossFunction, optimizer, iterations, DEVICE=DEVICE, verbose=verbose)
        else:
            results = nnutils.trainNetwork(net, trainloader, lossFunction, optimizer, iterations, DEVICE=DEVICE, verbose=verbose)
        
        # remove dropout and then measuring test performance
        storeDropout = results['net'].getDropout()
        results['net'].setDropout(0)
        results['trainloader'] = trainloader
        results['testloader'] = testloader
        results['pDropout'] = cDropout
        results['cWeightDecay'] = cWeightDecay
        testLoss, testAccuracy = nnutils.measurePerformance(results['net'], testloader, DEVICE=DEVICE, verbose=False)
        print(f"Network {runidx+1}/{numRuns} finished in {time.time()-t:.1f} seconds. pDropout:{cDropout:.1f}. WeightDecay:{cWeightDecay}. Test loss:{testLoss:.2f}. Test accuracy: {testAccuracy:.2f}%.")
        results['net'].setDropout(storeDropout)
        
        # Save Everything, delete unnecessary, continue
        fullResults.append(results)
        del net, results
    
    return fullResults

# Experiment to compare performance of eigenNets based on their initial state. 
def predictEigenNetPerformanceByInit(useNetID=None,pDropout=0.5,weightDecay=1e-3,iterations=10,learningRate=1e-2,dataset='MNIST',DEVICE=None,verbose=True):
    """
    # Experiment to compare performance of eigenNets based on their initial state
    # I want to do this with MNIST and CIFAR... 
    """
    
    # Check Experiment Parameters
    assert len(pDropout)==len(weightDecay)==len(useNetID), "pDropout, weightDecay, and useNetID have to be the same length"
    assert np.all(np.logical_or(useNetID==0, useNetID==1)), "useNetID can only include 0s and 1s, for eigenNet or eigenNet1"
    
    # Define useNet options
    useNet={0:eigModels.eigenNet,1:eigModels.eigenNet1}
    
    # First, select device to run networks on
    if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU = "cpu"
    print(f"Initiating experiment using {DEVICE}")
    
    # Prepare dataloaders 
    batchSize = 2**9
    if dataset=='MNIST':
        # In this experiment, the networks are all feedforward so we can use the same preprocessing stage for all of them
        preprocess = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
            transforms.Lambda(torch.flatten), # convert to vectors
        ])
        trainloader, testloader, numClasses = nnutils.downloadMNIST(batchSize=batchSize, preprocess=preprocess)
    else:
        raise ValueError("Dataset not recognized, must be 'MNIST' or 'CIFAR' (but cifar hasn't been coded yet)")
     
    lossFunction = nn.CrossEntropyLoss() # Note: this automatically applies softmax...
    actFunc = F.relu # To be used for activation function in first 3 layers (excluding output because we're using cross entropy loss)
    
    # Train Networks
    fullResults = []
    numRuns = len(pDropout)
    for runidx, prms in enumerate(zip(pDropout, weightDecay, useNetID)):
        cDropout, cWeightDecay, cUseNetID = prms
        
        t = time.time()
        
        # Create new network
        net = useNet[cUseNetID](trainloader,actFunc=actFunc,pDropout=cDropout,device=DEVICE) 
        net.to(DEVICE)
        
        # Set up optimizer based on learning rate
        if learningRate is None:
            optimizer = torch.optim.Adadelta(net.parameters(), weight_decay=cWeightDecay)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, weight_decay=cWeightDecay)
        
        # Train network
        results = nnutils.trainNetwork(net, trainloader, lossFunction, optimizer, iterations, DEVICE=DEVICE, verbose=verbose)
        
        # remove dropout and then measuring test performance
        storeDropout = results['net'].getDropout()
        results['net'].setDropout(0)
        results['trainloader'] = trainloader
        results['testloader'] = testloader
        results['pDropout'] = cDropout
        results['cWeightDecay'] = cWeightDecay
        testLoss, testAccuracy = nnutils.measurePerformance(results['net'], testloader, DEVICE=DEVICE, verbose=False)
        print(f"Network {runidx+1}/{numRuns} finished in {time.time()-t:.1f} seconds. pDropout:{cDropout:.1f}. WeightDecay:{cWeightDecay}. Test loss:{testLoss:.2f}. Test accuracy: {testAccuracy:.2f}%.")
        results['net'].setDropout(storeDropout)
        
        # Save Everything, delete unnecessary, continue
        fullResults.append(results)
        del net, results
    
    return fullResults

                                              


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import tools.alignmentAnalysisTools as aat
from torchvision import transforms


class CNN2P2(nn.Module):
    """
    CNN with 2 convolutional layers, a max pooling stage, and 2 feedforward layers
    Activation function is Relu by default (but can be chosen with hiddenactivation). 
    Output activation function is identity, because we're using CrossEntropyLoss
    """
    def __init__(self,weightvars=None,convActivation=F.relu,linearActivation=F.relu):
        super().__init__()
        self.numLayers = 4
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)      
        self.maxPool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(256, 256) #4608, 512       
        self.o = nn.Linear(256, 10) #512, 10
        
        self.convActivation=convActivation
        self.linearActivation=linearActivation
        
        if weightvars != None:
            # custom init 
            nn.init.normal_(self.conv1.weight,mean=0,std=weightvars[0]) 
            nn.init.normal_(self.conv2.weight,mean=0,std=weightvars[1])
            nn.init.normal_(self.fc1.weight,mean=0,std=weightvars[3])             
            nn.init.normal_(self.o.weight,mean=0,std=weightvars[4])  

    def forward(self, x):        
        self.c1 = self.convActivation(self.conv1(x))
        self.c2 = self.maxPool(self.convActivation(self.conv2(self.c1)))
        self.f1 = self.linearActivation(self.fc1(torch.flatten(self.c2,1)))
        self.out = self.o(self.f1)
        return self.out 
    
    def getActivations(self,x):
        out = self.forward(x)
        activations = []
        activations.append(self.c1)
        activations.append(self.c2)
        activations.append(self.f1)
        activations.append(self.out)
        return activations
    
    def getNetworkWeights(self):
        netWeights = []
        netWeights.append(self.conv1.weight.data.clone().detach())
        netWeights.append(self.conv2.weight.data.clone().detach())
        netWeights.append(self.fc1.weight.data.clone().detach())
        netWeights.append(self.o.weight.data.clone().detach())
        return netWeights
    
    def compareNetworkWeights(self, initWeights):
        currWeights = self.getNetworkWeights()
        deltaWeights = []
        for iw,cw in zip(initWeights,currWeights):
            iw = torch.flatten(iw,1)
            cw = torch.flatten(cw,1)
            deltaWeights.append(torch.norm(iw-cw,dim=1))
        return deltaWeights
        
    def measureSimilarity(self,x):
        activations = self.getActivations(x)            
        similarity = []
        similarity.append(torch.mean(aat.similarityConvLayer(x, self.conv1),axis=1))
        similarity.append(torch.mean(aat.similarityConvLayer(activations[0], self.conv2),axis=1))
        similarity.append(aat.similarityLinearLayer(torch.flatten(activations[1],1), self.fc1))
        similarity.append(aat.similarityLinearLayer(activations[2], self.o))
        return similarity
        
    def measureAlignment(self,x):
        activations = self.getActivations(x)            
        alignment = []
        alignment.append(torch.mean(aat.alignmentConvLayer(x, self.conv1),axis=1))
        alignment.append(torch.mean(aat.alignmentConvLayer(activations[0], self.conv2),axis=1))
        alignment.append(aat.alignmentLinearLayer(torch.flatten(activations[1],1), self.fc1))
        alignment.append(aat.alignmentLinearLayer(activations[2], self.o))
        return alignment
    
    @staticmethod
    def targetedDropout(net,x,idx=None,layer=None,returnFull=False):
        assert layer>=0 and layer<=2, "dropout only works on first three layers"
        c1 = net.convActivation(net.conv1(x))
        if layer==0: c1[:,idx]=0
        c2 = net.maxPool(net.convActivation(net.conv2(c1))) 
        if layer==1: c2[:,idx]=0
        f1 = net.linearActivation(net.fc1(torch.flatten(c2,1)))        
        if layer==2: f1[:,idx]=0
        out = net.o(f1)
        if returnFull:
            return c1,c2,f1,out
        else: 
            return out
    
    @staticmethod
    def avgFromFull(full):
        numEpochs = len(full)
        numLayers = len(full[0])
        avgFull = torch.zeros((numLayers,numEpochs))
        for layer in range(numLayers):
            avgFull[layer,:] = torch.tensor([torch.mean(f[layer]) for f in full])
        return avgFull.cpu()
    
    @staticmethod
    def layerFromFull(full,layer):
        return torch.cat([f[layer].reshape(-1,1) for f in full],dim=1).cpu()    
    
    
class MLP(nn.Module):
    '''
    Multilayer Perceptron for MNIST
    '''
    def __init__(self, hiddenLayerWidth=(100,)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784,hiddenLayerWidth[0]),
            nn.ReLU(),
        )
        for hiddenLayer in range(1,len(hiddenLayerWidth)):
            self.layers.append(nn.Linear(hiddenLayerWidth[hiddenLayer-1],hiddenLayerWidth[hiddenLayer]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hiddenLayerWidth[-1],10))
        self.layers.append(nn.Softmax())
            
    def forward(self, x):
        x = torch.flatten(x,1)
        return self.layers(x)


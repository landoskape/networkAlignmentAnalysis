import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from torch import nn
import tools.alignmentAnalysisTools as aat
from torchvision import models, transforms

class eigenNet(nn.Module):
    """
    MLP that learns on the weights of each eigenvector in the input activity of previous layers 
    Activation function is Relu by default (but can be chosen with hiddenactivation). 
    Output activation function is identity, because we're using CrossEntropyLoss
    """
    def __init__(self,dataloader,actFunc=F.relu,pDropout=0,device=None):
        super().__init__()
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.numLayers = 4
        # Initialize weights with standard method
        self.fc1 = nn.Linear(784,100) # Need a maximum of 784 weights for all possible eigenvectors of input
        self.fc2 = nn.Linear(100,100) # And same here
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,10)
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=pDropout)
        self.device = device
        self.dataloader = dataloader # keep dataloader for fitting eigenstructure of network
        self.to(self.device)
        self.fitEigenstructure()
    
    def fitEigenstructure(self):
        # Just calls getEigenstructure, but stores all arrays as object attributes
        self.eval0,self.evec0,self.eval1,self.evec1,self.eval2,self.evec2,self.eval3,self.evec3 = self.getEigenstructure()
        
    def getEigenstructure(self):
        # Get inputs, measure eigenstructure, get projections
        eigData = self.getDataFromDataLoader()
        eval0,evec0 = self.doEig(eigData.T)
        hidden1 = self.actFunc(self.fc1(self.getProjections(eval0,evec0,eigData)))
        eval1,evec1 = self.doEig(hidden1.T)
        hidden2 = self.actFunc(self.fc2(self.getProjections(eval1,evec1,hidden1)))
        eval2,evec2 = self.doEig(hidden2.T)
        hidden3 = self.actFunc(self.fc3(self.getProjections(eval2,evec2,hidden2)))
        eval3,evec3 = self.doEig(hidden3.T)
        return eval0,evec0,eval1,evec1,eval2,evec2,eval3,evec3
        
    def forward(self, x):
        self.hidden1 = self.actFunc(self.fc1(self.getProjections(self.eval0,self.evec0,x)))
        self.hidden2 = self.actFunc(self.fc2(self.dropout(self.getProjections(self.eval1,self.evec1,self.hidden1))))
        self.hidden3 = self.actFunc(self.fc3(self.dropout(self.getProjections(self.eval2,self.evec2,self.hidden2))))
        self.output = self.actFunc(self.fc4(self.dropout(self.getProjections(self.eval3,self.evec3,self.hidden3))))
        return self.output
        
    def getProjections(self,evals,evecs,batch):
        return torch.squeeze(torch.matmul(evecs.T[None,:,:], batch[:,:,None]),dim=2)
        
    def doEig(self,activations):
        ccov = torch.cov(activations)
        crank = torch.linalg.matrix_rank(ccov)
        evals,evecs = torch.linalg.eigh(ccov)
        evalIdx = torch.argsort(evals,descending=True) # descending eigenvalues
        evals = evals[evalIdx] # sort appropriately
        evecs = evecs[:,evalIdx] # sort
        evals[crank:] = 0 # remove numerical errors
        return evals.clone().detach(), evecs.clone().detach()
       
    def getDataFromDataLoader(self):
        return torch.cat([batch[0].to(self.device) for batch in self.dataloader]) # load data to be stored in network for measuring eigenstructure efficiently with minimal overhead
        
    def setDropout(self,pDropout):
        self.dropout.p = pDropout
    
    def getDropout(self):
        return self.dropout.p

    def getActivations(self,x):
        out = self.forward(x)
        activations = []
        activations.append(self.hidden1)
        activations.append(self.hidden2)
        activations.append(self.hidden3)
        activations.append(self.output)
        return activations
    
    def getNetworkWeights(self):
        netWeights = []
        netWeights.append(self.fc1.weight.data.clone().detach())
        netWeights.append(self.fc2.weight.data.clone().detach())
        netWeights.append(self.fc3.weight.data.clone().detach())
        netWeights.append(self.fc4.weight.data.clone().detach())
        return netWeights
    
    def compareNetworkWeights(self, initWeights):
        currWeights = self.getNetworkWeights()
        deltaWeights = []
        for iw,cw in zip(initWeights,currWeights):
            iw = torch.flatten(iw,1)
            cw = torch.flatten(cw,1)
            deltaWeights.append(torch.norm(cw-iw,dim=1))
        return deltaWeights
    
    def measureAlignment(self,x):
        alignment = []
        alignment.append(alignmentEigenLayer(self.eval0, self.fc1.weight.data.clone().detach()))
        alignment.append(alignmentEigenLayer(self.eval1, self.fc2.weight.data.clone().detach()))
        alignment.append(alignmentEigenLayer(self.eval2, self.fc3.weight.data.clone().detach()))
        alignment.append(alignmentEigenLayer(self.eval3, self.fc4.weight.data.clone().detach()))
        return alignment
    
    @staticmethod
    def avgFromFull(full):
        numEpochs = len(full)
        numLayers = len(full[0])
        avgFull = torch.zeros((numLayers,numEpochs))
        for layer in range(numLayers):
            avgFull[layer,:] = torch.tensor([torch.mean(f[layer]) for f in full])
        return avgFull.cpu()
    
    @staticmethod
    def layerFromFull(full,layer,dim=1):
        if dim==1: 
            return torch.cat([f[layer][:,None] for f in full],dim=dim).cpu() 
        elif dim==2:
            return torch.cat([f[layer][:,:,None] for f in full],dim=dim).cpu() 
        else:
            raise ValueError("Haven't coded layerFromFull for dimensions other than 1 or 2!")             
    

class eigenNet1(nn.Module):
    """
    MLP that learns on the weights of each eigenvector in the input activity of previous layer (only for first layer...) 
    Activation function is Relu by default (but can be chosen with hiddenactivation). 
    Output activation function is identity, because we're using CrossEntropyLoss
    """
    def __init__(self,dataloader,actFunc=F.relu,pDropout=0,device=None):
        super().__init__()
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.numLayers = 4
        # Initialize weights with standard method
        self.fc1 = nn.Linear(784,100) # Need a maximum of 784 weights for all possible eigenvectors of input
        self.fc2 = nn.Linear(100,100) # And same here
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,10)
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=pDropout)
        self.device = device
        self.to(device)
        self.dataloader = dataloader # keep dataloader for fitting eigenstructure of network
        self.eval0, self.evec0 = self.doEig(self.getDataFromDataLoader().T) # Fit eigenstructure of input data
        
    def forward(self, x):
        self.hidden1 = self.actFunc(self.fc1(self.getProjections(self.eval0,self.evec0,x)))
        self.hidden2 = self.actFunc(self.fc2(self.dropout(self.hidden1)))
        self.hidden3 = self.actFunc(self.fc3(self.dropout(self.hidden2)))
        self.output = self.actFunc(self.fc4(self.dropout(self.hidden3)))
        return self.output
        
    def getProjections(self,evals,evecs,batch):
        return torch.squeeze(torch.matmul(evecs.T[None,:,:], batch[:,:,None]),dim=2)
        
    def doEig(self,activations):
        ccov = torch.cov(activations)
        crank = torch.linalg.matrix_rank(ccov)
        evals,evecs = torch.linalg.eigh(ccov)
        evalIdx = torch.argsort(evals,descending=True) # descending eigenvalues
        evals = evals[evalIdx] # sort appropriately
        evecs = evecs[:,evalIdx] # sort
        evals[crank:] = 0 # remove numerical errors
        return evals.clone().detach(), evecs.clone().detach()
    
    def getDataFromDataLoader(self):
        return torch.cat([batch[0].to(self.device) for batch in self.dataloader]) # load data to be stored in network for measuring eigenstructure efficiently with minimal overhead
        
    def setDropout(self,pDropout):
        self.dropout.p = pDropout
    
    def getDropout(self):
        return self.dropout.p

    def getActivations(self,x):
        out = self.forward(x)
        activations = []
        activations.append(self.hidden1)
        activations.append(self.hidden2)
        activations.append(self.hidden3)
        activations.append(self.output)
        return activations
    
    def getNetworkWeights(self):
        netWeights = []
        netWeights.append(self.fc1.weight.data.clone().detach())
        netWeights.append(self.fc2.weight.data.clone().detach())
        netWeights.append(self.fc3.weight.data.clone().detach())
        netWeights.append(self.fc4.weight.data.clone().detach())
        return netWeights
    
    def compareNetworkWeights(self, initWeights):
        currWeights = self.getNetworkWeights()
        deltaWeights = []
        for iw,cw in zip(initWeights,currWeights):
            iw = torch.flatten(iw,1)
            cw = torch.flatten(cw,1)
            deltaWeights.append(torch.norm(cw-iw,dim=1))
        return deltaWeights
        
    def measureAlignment(self,x):
        activations = self.getActivations(x)            
        alignment = []
        alignment.append(alignmentEigenLayer(self.eval0, self.fc1.weight.data.clone().detach()))
        alignment.append(aat.alignmentLinearLayer(activations[0], self.fc2))
        alignment.append(aat.alignmentLinearLayer(activations[1], self.fc3))
        alignment.append(aat.alignmentLinearLayer(activations[2], self.fc4))
        return alignment
    
    @staticmethod
    def avgFromFull(full):
        numEpochs = len(full)
        numLayers = len(full[0])
        avgFull = torch.zeros((numLayers,numEpochs))
        for layer in range(numLayers):
            avgFull[layer,:] = torch.tensor([torch.mean(f[layer]) for f in full])
        return avgFull.cpu()
    
    @staticmethod
    def layerFromFull(full,layer,dim=1):
        if dim==1: 
            return torch.cat([f[layer][:,None] for f in full],dim=dim).cpu() 
        elif dim==2:
            return torch.cat([f[layer][:,:,None] for f in full],dim=dim).cpu() 
        else:
            raise ValueError("Haven't coded layerFromFull for dimensions other than 1 or 2!")       
    

def alignmentEigenLayer(evals,weights):
    # Weights should be an (out x in) array aligned to the respective eigenvalues in evals
    rq = torch.sum(evals[None,:] * weights**2, dim=1) / torch.sum(weights**2, dim=1)
    return rq/torch.sum(evals)
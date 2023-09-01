class alignmentModel(nn.Module)::
    """
    Skeleton for nn modules that utilize alignment functions
    Designed to automatically apply useful methods related to measuring alignment and dropout
    But you just have to structure the __init__ file correctly
    """
    def __init__(self,convActivation=F.relu,linearActivation=F.relu):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)      
        self.maxPool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(256, 256) #4608, 512       
        self.o = nn.Linear(256, 10) #512, 10

        self.layerRegistration = {
            'conv1':'conv',
            'conv2':'conv',
            'fc1':True,
            'o':True
        }

        self.layers = []
        self.layers.append(nn.Sequential(
            self.conv1,
            nn.ReLU(),
        ))

        self.layers.append(nn.Sequential(
            self.conv2,
            nn.ReLU(),
            self.maxPool
        ))
        self.layers.append(nn.Sequential(
            nn.Flatten(start_dim=1),
            self.fc1,
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(
            self.o
        ))

        self.numLayers = len(self.layers)

    def forward(self, x):        
        self.activations = [None]*self.numLayers
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            self.activations[idx]=x
        return x

    def getActivations(self,x):
        # if x has already been passed, can just return self.activations!
        out = self.forward(x)
        return self.activations
    
    def getNetworkWeights(self,onlyFF=False):
        netWeights = []
        if not onlyFF:
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
            deltaWeights.append(torch.norm(cw-iw,dim=1))
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
    
    def manualShape(self,evals,evecs,DEVICE,evalTransform=None):
        if evalTransform is None: evalTransform = lambda x:x
            
        sbetas = [] # produce signed betas
        netweights = self.getNetworkWeights(onlyFF=True)
        for evc,nw in zip(evecs,netweights):
            nw = nw / torch.norm(nw,dim=1,keepdim=True)
            sbetas.append(nw.cpu() @ evc)
        
        ffLayers = [2,3]
        shapedWeights = [[] for _ in range(len(ffLayers))]
        for idx in range(len(ffLayers)):
            assert np.all(evals[idx]>=0), "Found negative eigenvalues..."
            cFractionVariance = evals[idx]/np.sum(evals[idx]) # compute fraction of variance explained by each eigenvector
            cKeepFraction = evalTransform(cFractionVariance).astype(cFractionVariance.dtype) # make sure the datatype doesn't change, otherwise pytorch einsum will be unhappy
            assert np.all(cKeepFraction>=0), "Found negative transformed keep fractions. This means the transform function has an improper form." 
            assert np.all(cKeepFraction<=1), "Found keep fractions greater than 1. This is bad practice, design the evalTransform function to have a domain and range within [0,1]"
            weightNorms = torch.norm(netweights[idx],dim=1,keepdim=True) # measure norm of weights (this will be invariant to the change)
            evecComposition = torch.einsum('oi,xi->oxi',sbetas[idx],torch.tensor(evecs[idx])) # create tensor composed of each eigenvector scaled to it's contribution in each weight vector
            newComposition = torch.einsum('oxi,i->ox',evecComposition,torch.tensor(cKeepFraction)).to(DEVICE) # scale eigenvectors based on their keep fraction (by default scale them by their variance)
            shapedWeights[idx] = newComposition / torch.norm(newComposition,dim=1,keepdim=True) * weightNorms
        
        # Assign new weights to network
        self.fc1.weight.data = shapedWeights[0]
        self.o.weight.data = shapedWeights[1]
    
    @staticmethod
    def targetedDropout(net,x,idx=None,layer=None,returnFull=False):
        assert layer>=0 and layer<=2, "dropout only works on first three layers"
        c1 = net.convActivation(net.conv1(x))
        if layer==0: 
            fracDropout = len(idx)/c1.shape[1]
            c1[:,idx]=0
            c1 = c1 * (1 - fracDropout)
        c2 = net.maxPool(net.convActivation(net.conv2(c1))) 
        if layer==1: 
            fracDropout = len(idx)/c2.shape[1]
            c2[:,idx]=0
            c2 = c2 * (1 - fracDropout)
        f1 = net.linearActivation(net.fc1(torch.flatten(c2,1)))        
        if layer==2: 
            fracDropout = len(idx)/f1.shape[1]
            f1[:,idx]=0
            f1 = f1 * (1 - fracDropout)
        out = net.o(f1)
        if returnFull: return c1,c2,f1,out
        else: return out
    
    @staticmethod
    def mlTargetedDropout(net,x,idx,layer,returnFull=False):
        assert type(idx) is tuple and type(layer) is tuple, "idx and layer need to be tuples"
        assert len(idx)==len(layer), "idx and layer need to have the same length"
        npLayer = np.array(layer)
        assert len(npLayer)==len(np.unique(npLayer)), "layer must not have any repeated elements"
        # Do forward pass with targeted dropout
        c1 = net.convActivation(net.conv1(x))
        if np.any(npLayer==0):
            cIndex = idx[npLayer==0]
            fracDropout=len(cIndex)/c1.shape[1]
            c1[:,cIndex]=0
            c1 = c1 * (1 - fracDropout)
        c2 = net.maxPool(net.convAcivation(net.conv2(c1)))
        if np.any(npLayer==1):
            cIndex = idx[npLayer==1]
            fracDropout=len(cIndex)/c2.shape[1]
            c2[:,cIndex]=0
            c2 = c21 * (1 - fracDropout)
        f1 = net.linearActivation(net.fc1(torch.flatten(c2,1)))
        if np.any(npLayer==2):
            cIndex = idx[npLayer==2]
            fracDropout = len(cIndex)/f1.shape[1]
            f1[:,cIndex]=0
            f1 = f1 * (1 - fracDropout)
        out = net.o(f1)
        if returnFull: return c1,c2,f1,out
        else: return out
    
    @staticmethod
    def inputEigenfeatures(net, dataloader, onlyFF=True, DEVICE=None):
        # Handle DEVICE if not provided
        if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Measure Activations (without dropout) for all images
        storeDropout = net.getDropout()
        net.setDropout(0) # no dropout for measuring eigenfeatures
        allimages = []
        activations = []
        for images, label in dataloader:    
            allimages.append(images)
            images = images.to(DEVICE)
            label = label.to(DEVICE)
            activations.append(net.getActivations(images))
        net.setDropout(storeDropout)
        
        # Consolidate variable structure
        allinputs = []
        if not onlyFF:
            # Only add inputs to convolutional layers if onlyFF switch is off
            allinputs.append(torch.flatten(torch.cat(allimages,dim=0).detach().cpu(),1)) # inputs to first convolutional layer
            allinputs.append(torch.flatten(torch.cat([cact[0] for cact in activations],dim=0).detach().cpu(),1)) # inputs to second convolutional layer
        allinputs.append(torch.flatten(torch.cat([cact[1] for cact in activations],dim=0).detach().cpu(),1)) # inputs to first feedforward layer
        allinputs.append(torch.cat([cact[2] for cact in activations],dim=0).detach().cpu()) # inputs to last convolutional layer
            
        # Measure eigenfeatures for input to each feedforward layer
        eigenvalues = []
        eigenvectors = []
        for ai in allinputs:
            # Covariance matrix is positive semidefinite, but numerical errors can produce negative eigenvalues
            ccov = torch.cov(ai.T)
            crank = torch.linalg.matrix_rank(ccov)
            w,v = sp.linalg.eigh(ccov)
            widx = np.argsort(w)[::-1]
            w = w[widx]
            v = v[:,widx]
            # Automatically set eigenvalues to 0 when they are numerical errors!
            w[crank:]=0
            eigenvalues.append(w)
            eigenvectors.append(v)
            
        return eigenvalues, eigenvectors
    
    @staticmethod
    def measureEigenFeatures(net, dataloader, onlyFF=True, DEVICE=None):
        eigenvalues,eigenvectors = CNN2P2.inputEigenfeatures(net, dataloader, onlyFF=onlyFF, DEVICE=DEVICE)
        
        # Measure dot product of weights on eigenvectors for each layer
        beta = []
        netweights = net.getNetworkWeights(onlyFF=onlyFF)
        for evc,nw in zip(eigenvectors,netweights):
            nw = nw / torch.norm(nw,dim=1,keepdim=True)
            beta.append(torch.abs(nw.cpu() @ evc))
            
        return beta, eigenvalues, eigenvectors
    
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




















    
    
class CNN2P2(nn.Module):
    """
    CNN with 2 convolutional layers, a max pooling stage, and 2 feedforward layers
    Activation function is Relu by default (but can be chosen with hiddenactivation). 
    Output activation function is identity, because we're using CrossEntropyLoss
    """
    def __init__(self,convActivation=F.relu,linearActivation=F.relu):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)      
        self.maxPool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(256, 256) #4608, 512       
        self.o = nn.Linear(256, 10) #512, 10

        self.layers = []
        self.layers.append(nn.Sequential(
            self.conv1,
            nn.ReLU(),
        ))

        self.layers.append(nn.Sequential(
            self.conv2,
            nn.ReLU(),
            self.maxPool
        ))
        self.layers.append(nn.Sequential(
            nn.Flatten(start_dim=1),
            self.fc1,
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(
            self.o
        ))

        self.numLayers = len(self.layers)

    def forward(self, x):        
        self.activations = [None]*self.numLayers
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            self.activations[idx]=x
        return x
        
    def getActivations(self,x=None,fromStored=False):
        if fromStored: assert len(self.activations
        out = self.forward(x)
        activations = []
        activations.append(self.c1)
        activations.append(self.c2)
        activations.append(self.f1)
        activations.append(self.out)
        return activations
    
    def getNetworkWeights(self,onlyFF=False):
        netWeights = []
        if not onlyFF:
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
            deltaWeights.append(torch.norm(cw-iw,dim=1))
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
    
    def manualShape(self,evals,evecs,DEVICE,evalTransform=None):
        if evalTransform is None: evalTransform = lambda x:x
            
        sbetas = [] # produce signed betas
        netweights = self.getNetworkWeights(onlyFF=True)
        for evc,nw in zip(evecs,netweights):
            nw = nw / torch.norm(nw,dim=1,keepdim=True)
            sbetas.append(nw.cpu() @ evc)
        
        ffLayers = [2,3]
        shapedWeights = [[] for _ in range(len(ffLayers))]
        for idx in range(len(ffLayers)):
            assert np.all(evals[idx]>=0), "Found negative eigenvalues..."
            cFractionVariance = evals[idx]/np.sum(evals[idx]) # compute fraction of variance explained by each eigenvector
            cKeepFraction = evalTransform(cFractionVariance).astype(cFractionVariance.dtype) # make sure the datatype doesn't change, otherwise pytorch einsum will be unhappy
            assert np.all(cKeepFraction>=0), "Found negative transformed keep fractions. This means the transform function has an improper form." 
            assert np.all(cKeepFraction<=1), "Found keep fractions greater than 1. This is bad practice, design the evalTransform function to have a domain and range within [0,1]"
            weightNorms = torch.norm(netweights[idx],dim=1,keepdim=True) # measure norm of weights (this will be invariant to the change)
            evecComposition = torch.einsum('oi,xi->oxi',sbetas[idx],torch.tensor(evecs[idx])) # create tensor composed of each eigenvector scaled to it's contribution in each weight vector
            newComposition = torch.einsum('oxi,i->ox',evecComposition,torch.tensor(cKeepFraction)).to(DEVICE) # scale eigenvectors based on their keep fraction (by default scale them by their variance)
            shapedWeights[idx] = newComposition / torch.norm(newComposition,dim=1,keepdim=True) * weightNorms
        
        # Assign new weights to network
        self.fc1.weight.data = shapedWeights[0]
        self.o.weight.data = shapedWeights[1]
    
    @staticmethod
    def targetedDropout(net,x,idx=None,layer=None,returnFull=False):
        assert layer>=0 and layer<=2, "dropout only works on first three layers"
        c1 = net.convActivation(net.conv1(x))
        if layer==0: 
            fracDropout = len(idx)/c1.shape[1]
            c1[:,idx]=0
            c1 = c1 * (1 - fracDropout)
        c2 = net.maxPool(net.convActivation(net.conv2(c1))) 
        if layer==1: 
            fracDropout = len(idx)/c2.shape[1]
            c2[:,idx]=0
            c2 = c2 * (1 - fracDropout)
        f1 = net.linearActivation(net.fc1(torch.flatten(c2,1)))        
        if layer==2: 
            fracDropout = len(idx)/f1.shape[1]
            f1[:,idx]=0
            f1 = f1 * (1 - fracDropout)
        out = net.o(f1)
        if returnFull: return c1,c2,f1,out
        else: return out
    
    @staticmethod
    def mlTargetedDropout(net,x,idx,layer,returnFull=False):
        assert type(idx) is tuple and type(layer) is tuple, "idx and layer need to be tuples"
        assert len(idx)==len(layer), "idx and layer need to have the same length"
        npLayer = np.array(layer)
        assert len(npLayer)==len(np.unique(npLayer)), "layer must not have any repeated elements"
        # Do forward pass with targeted dropout
        c1 = net.convActivation(net.conv1(x))
        if np.any(npLayer==0):
            cIndex = idx[npLayer==0]
            fracDropout=len(cIndex)/c1.shape[1]
            c1[:,cIndex]=0
            c1 = c1 * (1 - fracDropout)
        c2 = net.maxPool(net.convAcivation(net.conv2(c1)))
        if np.any(npLayer==1):
            cIndex = idx[npLayer==1]
            fracDropout=len(cIndex)/c2.shape[1]
            c2[:,cIndex]=0
            c2 = c21 * (1 - fracDropout)
        f1 = net.linearActivation(net.fc1(torch.flatten(c2,1)))
        if np.any(npLayer==2):
            cIndex = idx[npLayer==2]
            fracDropout = len(cIndex)/f1.shape[1]
            f1[:,cIndex]=0
            f1 = f1 * (1 - fracDropout)
        out = net.o(f1)
        if returnFull: return c1,c2,f1,out
        else: return out
    
    @staticmethod
    def inputEigenfeatures(net, dataloader, onlyFF=True, DEVICE=None):
        # Handle DEVICE if not provided
        if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Measure Activations (without dropout) for all images
        storeDropout = net.getDropout()
        net.setDropout(0) # no dropout for measuring eigenfeatures
        allimages = []
        activations = []
        for images, label in dataloader:    
            allimages.append(images)
            images = images.to(DEVICE)
            label = label.to(DEVICE)
            activations.append(net.getActivations(images))
        net.setDropout(storeDropout)
        
        # Consolidate variable structure
        allinputs = []
        if not onlyFF:
            # Only add inputs to convolutional layers if onlyFF switch is off
            allinputs.append(torch.flatten(torch.cat(allimages,dim=0).detach().cpu(),1)) # inputs to first convolutional layer
            allinputs.append(torch.flatten(torch.cat([cact[0] for cact in activations],dim=0).detach().cpu(),1)) # inputs to second convolutional layer
        allinputs.append(torch.flatten(torch.cat([cact[1] for cact in activations],dim=0).detach().cpu(),1)) # inputs to first feedforward layer
        allinputs.append(torch.cat([cact[2] for cact in activations],dim=0).detach().cpu()) # inputs to last convolutional layer
            
        # Measure eigenfeatures for input to each feedforward layer
        eigenvalues = []
        eigenvectors = []
        for ai in allinputs:
            # Covariance matrix is positive semidefinite, but numerical errors can produce negative eigenvalues
            ccov = torch.cov(ai.T)
            crank = torch.linalg.matrix_rank(ccov)
            w,v = sp.linalg.eigh(ccov)
            widx = np.argsort(w)[::-1]
            w = w[widx]
            v = v[:,widx]
            # Automatically set eigenvalues to 0 when they are numerical errors!
            w[crank:]=0
            eigenvalues.append(w)
            eigenvectors.append(v)
            
        return eigenvalues, eigenvectors
    
    @staticmethod
    def measureEigenFeatures(net, dataloader, onlyFF=True, DEVICE=None):
        eigenvalues,eigenvectors = CNN2P2.inputEigenfeatures(net, dataloader, onlyFF=onlyFF, DEVICE=DEVICE)
        
        # Measure dot product of weights on eigenvectors for each layer
        beta = []
        netweights = net.getNetworkWeights(onlyFF=onlyFF)
        for evc,nw in zip(eigenvectors,netweights):
            nw = nw / torch.norm(nw,dim=1,keepdim=True)
            beta.append(torch.abs(nw.cpu() @ evc))
            
        return beta, eigenvalues, eigenvectors
    
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
     
    
class MLP4(nn.Module):
    """
    MLP with 4 layers
    Activation function is Relu by default (but can be chosen with hiddenactivation). 
    Output activation function is identity, because we're using CrossEntropyLoss
    """
    def __init__(self,actFunc=F.relu,pDropout=0.5):
        super().__init__()
        self.numLayers = 4
        self.fc1 = nn.Linear(784,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,10)
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=pDropout)

    def forward(self, x):
        self.hidden1 = self.actFunc(self.fc1(x))
        self.hidden2 = self.actFunc(self.fc2(self.dropout(self.hidden1)))
        self.hidden3 = self.actFunc(self.fc3(self.dropout(self.hidden2)))
        self.output = self.fc4(self.dropout(self.hidden3))
        return self.output 
    
    def setDropout(self,pDropout):
        self.dropout.p = pDropout
    
    def getDropout(self):
        return self.dropout.p
    
    def doOjaUpdate(self, x, alpha):
        # Rule: dW = alpha * (xy - wy**2) 
        B = x.shape[0]
        activations = self.getActivations(x)
        # Layer 1:
        H,D = (activations[0].shape[1], x.shape[1])
        dfc1 = alpha * (activations[0].T @ x - torch.sum(self.fc1.weight.data.clone().detach().reshape(H,D,1) * (activations[0]*2).T.reshape(H,B,1).permute(0,2,1),dim=2))
        self.fc1.weight.data = self.fc1.weight.data + dfc1
        self.fc1.weight.data = self.fc1.weight.data / torch.norm(self.fc1.weight.data,dim=1,keepdim=True)
        #print(f"fc1: Weight.shape:{self.fc1.weight.data.shape}, update.shape:{dfc1.shape}")
        # Layer 2:
        H,D = (activations[1].shape[1], activations[0].shape[1])
        dfc2 = alpha * (activations[1].T @ activations[0] - torch.sum(self.fc2.weight.data.clone().detach().reshape(H,D,1) * (activations[1]*2).T.reshape(H,B,1).permute(0,2,1),dim=2))
        self.fc2.weight.data = self.fc2.weight.data + dfc2
        self.fc2.weight.data = self.fc2.weight.data / torch.norm(self.fc2.weight.data,dim=1,keepdim=True)
        #print(f"fc2: Weight.shape:{self.fc2.weight.data.shape}, update.shape:{dfc2.shape}")
        # Layer 3:
        H,D = (activations[2].shape[1], activations[1].shape[1])
        dfc3 = alpha * (activations[2].T @ activations[1] - torch.sum(self.fc3.weight.data.clone().detach().reshape(H,D,1) * (activations[2]*2).T.reshape(H,B,1).permute(0,2,1),dim=2))
        self.fc3.weight.data = self.fc3.weight.data + dfc3
        self.fc3.weight.data = self.fc3.weight.data / torch.norm(self.fc3.weight.data,dim=1,keepdim=True)
        #print(f"fc3: Weight.shape:{self.fc3.weight.data.shape}, update.shape:{dfc3.shape}")
        # Layer 4:
        H,D = (activations[3].shape[1], activations[2].shape[1])
        dfc4 = alpha * (activations[3].T @ activations[2] - torch.sum(self.fc4.weight.data.clone().detach().reshape(H,D,1) * (activations[3]*2).T.reshape(H,B,1).permute(0,2,1),dim=2))
        self.fc4.weight.data = self.fc4.weight.data + dfc4
        self.fc4.weight.data = self.fc4.weight.data / torch.norm(self.fc4.weight.data,dim=1,keepdim=True)
        #print(f"fc4: Weight.shape:{self.fc4.weight.data.shape}, update.shape:{dfc4.shape}")
        
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
    
    def measureSimilarity(self,x):
        activations = self.getActivations(x)            
        similarity = []
        similarity.append(aat.similarityLinearLayer(x, self.fc1))
        similarity.append(aat.similarityLinearLayer(activations[0], self.fc2))
        similarity.append(aat.similarityLinearLayer(activations[1], self.fc3))
        similarity.append(aat.similarityLinearLayer(activations[2], self.fc4))
        return similarity
        
    def measureAlignment(self,x):
        activations = self.getActivations(x)            
        alignment = []
        alignment.append(aat.alignmentLinearLayer(x, self.fc1))
        alignment.append(aat.alignmentLinearLayer(activations[0], self.fc2))
        alignment.append(aat.alignmentLinearLayer(activations[1], self.fc3))
        alignment.append(aat.alignmentLinearLayer(activations[2], self.fc4))
        return alignment
    
    def manualShape(self,evals,evecs,DEVICE,evalTransform=None):
        if evalTransform is None:
            evalTransform = lambda x:x
            
        sbetas = [] # produce signed betas
        netweights = self.getNetworkWeights()
        for evc,nw in zip(evecs,netweights):
            nw = nw / torch.norm(nw,dim=1,keepdim=True)
            sbetas.append(nw.cpu() @ evc)
        
        shapedWeights = [[] for _ in range(self.numLayers)]
        for layer in range(self.numLayers):
            assert np.all(evals[layer]>=0), "Found negative eigenvalues..."
            cFractionVariance = evals[layer]/np.sum(evals[layer]) # compute fraction of variance explained by each eigenvector
            cKeepFraction = evalTransform(cFractionVariance).astype(cFractionVariance.dtype) # make sure the datatype doesn't change, otherwise pytorch einsum will be unhappy
            assert np.all(cKeepFraction>=0), "Found negative transformed keep fractions. This means the transform function has an improper form." 
            assert np.all(cKeepFraction<=1), "Found keep fractions greater than 1. This is bad practice, design the evalTransform function to have a domain and range within [0,1]"
            weightNorms = torch.norm(netweights[layer],dim=1,keepdim=True) # measure norm of weights (this will be invariant to the change)
            evecComposition = torch.einsum('oi,xi->oxi',sbetas[layer],torch.tensor(evecs[layer])) # create tensor composed of each eigenvector scaled to it's contribution in each weight vector
            newComposition = torch.einsum('oxi,i->ox',evecComposition,torch.tensor(cKeepFraction)).to(DEVICE) # scale eigenvectors based on their keep fraction (by default scale them by their variance)
            shapedWeights[layer] = newComposition / torch.norm(newComposition,dim=1,keepdim=True) * weightNorms
        
        # Assign new weights to network
        self.fc1.weight.data = shapedWeights[0]
        self.fc2.weight.data = shapedWeights[1]
        self.fc3.weight.data = shapedWeights[2]
        self.fc4.weight.data = shapedWeights[3]
        
    
    @staticmethod
    def targetedDropout(net,x,idx=None,layer=None,returnFull=False):
        assert layer>=0 and layer<=2, "dropout only works on first three layers"
        h1 = net.actFunc(net.fc1(x))
        if layer==0: 
            fracDropout = len(idx)/h1.shape[1]
            h1[:,idx]=0
            h1 = h1 * (1 - fracDropout)
        h2 = net.actFunc(net.fc2(h1))
        if layer==1: 
            fracDropout = len(idx)/h2.shape[1]
            h2[:,idx]=0
            h2 = h2 * (1 - fracDropout)            
        h3 = net.actFunc(net.fc3(h2))
        if layer==2: 
            fracDropout = len(idx)/h3.shape[1]
            h3[:,idx]=0
            h3 = h3 * (1 - fracDropout)
        out = net.fc4(h3)
        if returnFull: return h1,h2,h3,out
        else: return out
    
    @staticmethod
    def mlTargetedDropout(net,x,idx,layer,returnFull=False):
        assert type(idx) is tuple and type(layer) is tuple, "idx and layer need to be tuples"
        assert len(idx)==len(layer), "idx and layer need to have the same length"
        npLayer = np.array(layer)
        assert len(npLayer)==len(np.unique(npLayer)), "layer must not have any repeated elements"
        # Do forward pass with targeted dropout
        h1 = net.actFunc(net.fc1(x))
        if np.any(npLayer==0):
            cIndex = idx[npLayer==0]
            fracDropout = len(cIndex)/h1.shape[1]
            h1[:,cIndex]=0
            h1 = h1 * (1 - fracDropout)
        h2 = net.actFunc(net.fc2(h1))
        if np.any(npLayer==1):
            cIndex = idx[npLayer==1]
            fracDropout = len(cIndex)/h2.shape[1]
            h2[:,cIndex]=0
            h2 = h2 * (1 - fracDropout)            
        h3 = net.actFunc(net.fc3(h2))
        if np.any(npLayer==2):
            cIndex = idx[npLayer==2]
            fracDropout = len(cIndex)/h3.shape[1]
            h3[:,cIndex]=0
            h3 = h3 * (1 - fracDropout)
        out = net.fc4(h3)
        if returnFull: return h1,h2,h3,out
        else: return out
    
    @staticmethod
    def measureEigenFeatures(net, dataloader, DEVICE=None):
        # Handle DEVICE if not provided
        if DEVICE is None: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Measure Activations (without dropout) for all images
        storeDropout = net.getDropout()
        net.setDropout(0) # no dropout for measuring eigenfeatures
        allimages = []
        activations = []
        for images, label in dataloader:    
            allimages.append(images)
            images = images.to(DEVICE)
            label = label.to(DEVICE)
            activations.append(net.getActivations(images))
        net.setDropout(storeDropout)

        # Consolidate variable structure
        NL = net.numLayers
        allinputs = []
        allinputs.append(torch.cat(allimages,dim=0).detach().cpu())
        for layer in range(NL-1):
            allinputs.append(torch.cat([cact[layer] for cact in activations],dim=0).detach().cpu())

        # Measure eigenfeatures for each layer
        eigenvalues = []
        eigenvectors = []
        for ai in allinputs:
            # Covariance matrix is positive semidefinite, but numerical errors can produce negative eigenvalues
            ccov = torch.cov(ai.T)
            crank = torch.linalg.matrix_rank(ccov)
            w,v = sp.linalg.eigh(ccov)
            widx = np.argsort(w)[::-1]
            w = w[widx]
            v = v[:,widx]
            # Automatically set eigenvalues to 0 when they are numerical errors!
            w[crank:]=0
            eigenvalues.append(w)
            eigenvectors.append(v)

        # Measure dot product of weights on eigenvectors for each layer
        beta = []
        netweights = net.getNetworkWeights()
        for evc,nw in zip(eigenvectors,netweights):
            nw = nw / torch.norm(nw,dim=1,keepdim=True)
            beta.append(torch.abs(nw.cpu() @ evc))
            
        return beta, eigenvalues, eigenvectors
    
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
            
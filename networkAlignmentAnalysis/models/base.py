

def avgFromFull(full):
    numEpochs = len(full)
    numLayers = len(full[0])
    avgFull = torch.zeros((numLayers,numEpochs))
    for layer in range(numLayers):
        avgFull[layer,:] = torch.tensor([torch.mean(f[layer]) for f in full])
    return avgFull.cpu()

def layerFromFull(full,layer,dim=1):
    if dim==1: 
        return torch.cat([f[layer][:,None] for f in full],dim=dim).cpu() 
    elif dim==2:
        return torch.cat([f[layer][:,:,None] for f in full],dim=dim).cpu() 
    else:
        raise ValueError("Haven't coded layerFromFull for dimensions other than 1 or 2!")





def ExperimentalNetwork(AlignmentNetwork):
    """maintain some experimental methods here"""
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
    
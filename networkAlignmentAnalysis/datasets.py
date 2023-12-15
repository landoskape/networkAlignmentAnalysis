from . import files

import torch
import torchvision

"""
Should turn these into classes for loading...
Also I want the measure performance method to be a part of a standard dataloader class
"""

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



def downloadMNIST(batchSize=1000, preprocess=None):
    dataPath = files.getDataPath('MNIST')
    trainset = torchvision.datasets.MNIST(root=dataPath, train=True, download=True, transform=preprocess)
    testset = torchvision.datasets.MNIST(root=dataPath, train=False, download=True, transform=preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True, num_workers=2)
    return trainloader, testloader

def downloadImageNet(batchSize=1000):
    dataPath = files.getDataPath('ImageNet')
    valTransform = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
    # valTransform = transforms.Compose([
    #     transforms.Resize(256,interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    #         torchvision.transforms.CenterCrop(224),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    #     ])
    valData = torchvision.datasets.ImageNet(dataPath,split='val',transform=valTransform)
    valLoader = torch.utils.data.DataLoader(valData, batch_size=500, shuffle=True, num_workers=2, pin_memory=False)
    return valLoader


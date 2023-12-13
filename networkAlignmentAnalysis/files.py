import os

def getDataPath(dataset='MNIST'):
    # Path to stored datasets (might add input argument for running on a different computer...)
    if dataset=='MNIST':
        return os.path.join('C:/', 'Users','andrew','Documents','machineLearning','datasets')
    elif dataset=='ImageNet':
        return os.path.join('C:/', 'Users','andrew','Documents','machineLearning','datasets','imagenet')
    else: 
        raise ValueError("Didn't recognize dataset string.")
from pathlib import Path

def local_path():
    return Path('C:/Users/andrew/Documents/machineLearning')

def data_path():
    return local_path() / 'datasets'

def dataset_path(dataset):
    """path to stored dataset"""
    if dataset=='MNIST':
        # 
        return data_path()

def getDataPath(dataset='MNIST'):
    # Path to stored datasets (might add input argument for running on a different computer...)
    if dataset=='MNIST':
        return os.path.join('C:/', 'Users','andrew','Documents','machineLearning','datasets')
    elif dataset=='ImageNet':
        return os.path.join('C:/', 'Users','andrew','Documents','machineLearning','datasets','imagenet')
    else: 
        raise ValueError("Didn't recognize dataset string.")
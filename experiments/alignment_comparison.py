import sys
import os

# add path that contains the alignment package
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import time
from tqdm import tqdm
import torch
from argparse import ArgumentParser

from matplotlib import pyplot as plt

from networkAlignmentAnalysis import files
from networkAlignmentAnalysis.models.registry import get_model
from networkAlignmentAnalysis.datasets import get_dataset
from networkAlignmentAnalysis import train
from networkAlignmentAnalysis.utils import avg_from_full

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = ArgumentParser(description='alignment_comparison')
    
    parser.add_argument('--network', type=str, default='MLP') # what base network architecture to use
    parser.add_argument('--dataset', type=str, default='MNIST') # what dataset to use

    parser.add_argument('--comparison', type=str, default='lr') # what comparison to do
    parser.add_argument('--lrs', type=float, nargs='*', default=[1e-3, 5e-4, 1e-4]) # which learning rates to use

    parser.add_argument('--epochs', type=int, default=1) # how many rounds of training to do
    parser.add_argument('--replicates', type=int, default=1) # how many copies of identical networks to train
    return parser.parse_args()

def load_networks(args):
    """
    method for loading networks

    depending on the experiment parameters (which comparison, which metaparams etc)
    this method will create multiple networks with requested parameters and return
    their optimizers and a params dictionary with the experiment parameters associated
    with each network
    """
    model_constructor = get_model(args.network)

    # compare learning rates
    if args.comparison == 'lr':
        lrs = [lr for lr in args.lrs for _ in range(args.replicates)]
        nets = [model_constructor() for _ in lrs]
        nets = [net.to(DEVICE) for net in nets]
        optimizers = [torch.optim.Adam(net.parameters(), lr=lr) for net, lr in zip(nets, lrs)]
        prms = {
            'lrs': lrs,
        }
        return nets, optimizers, prms
    
    else:
        raise ValueError(f"Comparison={args.comparision} is not recognized")
    
def load_dataset(args, transform_parameters):
    """supporting method for loading the requested dataset"""
    dataset_constructor = get_dataset(args.dataset)
    return dataset_constructor(transform_parameters=transform_parameters)

def train_networks(args):
    print('using device: ', DEVICE)
    
    # get networks
    nets, optimizers, prms = load_networks(args)

    # load dataset
    dataset = load_dataset(args, nets[0].get_transform_parameters(args.dataset))

    # do training loop
    parameters = dict(
        train_set=True,
        num_epochs=args.epochs,
        alignment=True,
        delta_weights=True,
    )
    train_results = train.train(nets, optimizers, dataset, **parameters)
    test_results = train.test(nets, dataset, **parameters)

    return train_results, test_results, prms

def plot_train_results(train_results, test_results, prms):
    num_train = train_results['loss'].size(0)
    align_mean = [avg_from_full(align).T for align in train_results['alignment']]

    figdim = 3
    fig, ax = plt.subplots(1, 3, figsize=(3*figdim, figdim), layout='constrained')
    ax[0].plot(range(num_train), train_results['loss'])
    ax[1].plot(range(num_train), train_results['performance'])
    ax[2].plot(range(num_train), align_mean[0])

    plt.show()

if __name__ == '__main__':

    # run parameters
    args = get_args()

    # do training 
    train_results, test_results, prms = train_networks(args)

    # plot
    plot_train_results(train_results, test_results, prms)
    
    



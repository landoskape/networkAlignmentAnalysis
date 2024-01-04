import sys
import os

# add path that contains the alignment package
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datetime import datetime
from copy import copy
from functools import partial
import numpy as np
from tqdm import tqdm
import torch
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import matplotlib as mpl

from networkAlignmentAnalysis.experiments.experiment import Experiment
from networkAlignmentAnalysis.models.registry import get_model
from networkAlignmentAnalysis.datasets import get_dataset
from networkAlignmentAnalysis import train
from networkAlignmentAnalysis.utils import avg_value_by_layer, compute_stats_by_type, transpose_list, named_transpose

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class OptimizeTransfer(Experiment):
    def get_basename(self):
        return 'optimize_transfer'
    
    def prepare_path(self):
        return [self.args.comparison, self.args.network, self.args.dataset]
    
    def make_args(self, parser):
        parser.add_argument('--network', type=str, default='MLP') # what base network architecture to use
        parser.add_argument('--dataset', type=str, default='MNIST') # what dataset to use

        # main experiment parameters

        # supporting parameters experiments

        # default parameters (if not controlled by the experiment parameters)
        
        # some metaparameters for the experiment
        parser.add_argument('--epochs', type=int, default=100) # how many rounds of training to do
        parser.add_argument('--replicates', type=int, default=10) # how many copies of identical networks to train
        
        # return parser
        return parser


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
            'lrs': lrs, # the value of the independent variable for each network
            'name': 'lr', # the name of the parameter being varied
            'vals': args.lrs, # the list of unique values for the relevant parameter
        }
        return nets, optimizers, prms
    
    else:
        raise ValueError(f"Comparison={args.comparision} is not recognized")
    
def load_dataset(args, transform_parameters):
    """supporting method for loading the requested dataset"""
    dataset_constructor = get_dataset(args.dataset)
    return dataset_constructor(transform_parameters=transform_parameters)

def train_networks(nets, optimizers, dataset, args):
    print('using device: ', DEVICE)
    
    # do training loop
    parameters = dict(
        train_set=True,
        num_epochs=args.epochs,
        alignment=True,
        delta_weights=True,
    )
    print('training networks...')
    train_results = train.train(nets, optimizers, dataset, **parameters)
    print('testing networks...')
    parameters['train_set'] = False
    test_results = train.test(nets, dataset, **parameters)

    return train_results, test_results, prms

if __name__ == '__main__':

     # Create experiment 
    exp = OptimizeTransfer()

    # get networks
    nets, optimizers, prms = load_networks(exp)

    # load dataset
    standard_transform = nets[0].get_transform_parameters(exp.args.dataset)
    dataset = load_dataset(exp, standard_transform)

    # do training 
    train_results, test_results, prms = train_networks(nets, optimizers, dataset, exp)

    # --- test networks at current stage of training on permuted MNIST ---
    def permute_pixels(batch, permute_idx=None):
        batch[0] = batch[0][:, permute_idx] # permute randomly on pixel dimension
        return batch
    
    # --- add extra transform to be done during "unwrap_batch" stage (implemented by DataSet)
    permute_transform = copy(standard_transform)
    permute_transform['extra_transform'] = partial(permute_pixels, permute_idx=torch.randperm(784))
    permute_dataset = load_dataset(exp, permute_transform)

    # plot results -- will need to rewrite this -- 
    # plot_train_results(train_results, test_results, prms)
    
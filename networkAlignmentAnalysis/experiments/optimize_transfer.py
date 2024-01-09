from copy import copy
from functools import partial
import torch

from .experiment import Experiment
from ..models.registry import get_model
from ..datasets import get_dataset
from .. import train
from ..utils import avg_value_by_layer, compute_stats_by_type, transpose_list, named_transpose

class OptimizeTransfer(Experiment):
    def get_basename(self):
        """defines a 'basename' for this experiment which is used for the results directory"""
        return 'optimize_transfer'
    
    def prepare_path(self):
        """
        defines a directory format to save each experiment on a path defined by the main
        experiment parameters
        """
        return [self.args.comparison, self.args.network, self.args.dataset, self.args.optimizer]
    
    def make_args(self, parser):
        """
        Method for adding experiment specific arguments to the argument parser
        """
        # Network & Dataset
        parser.add_argument('--network', type=str, default='MLP') # what base network architecture to use
        parser.add_argument('--dataset', type=str, default='MNIST') # what dataset to use
        parser.add_argument('--optimizer', type=str, default='SGD') # what optimizer to train with

        # main experiment parameters
        

        # supporting parameters for the experiments
        

        # some metaparameters for the experiment
        parser.add_argument('--epochs', type=int, default=100) # how many rounds of training to do
        parser.add_argument('--replicates', type=int, default=10) # how many copies of identical networks to train
        
        # return parser
        return parser
    
    def main(self):
        """
        main experiment loop
        """
        
        # get networks
        nets, optimizers, prms = self.load_networks()

        # load dataset
        standard_transform = nets[0].get_transform_parameters(self.args.dataset)
        dataset = self.load_dataset(standard_transform)

        # do training (probably...)
        # train_results, test_results = self.train_networks(nets, optimizers, dataset)

        # --- test networks at current stage of training on permuted MNIST ---
        # --- (permuting pixels just an idea for transfer learning!!!) ---
        def permute_pixels(batch, permute_idx=None):
            batch[0] = batch[0][:, permute_idx] # permute randomly on pixel dimension
            return batch
        
        # --- add extra transform to be done during "unwrap_batch" stage (implemented by DataSet)
        permute_transform = copy(standard_transform)
        permute_transform['extra_transform'] = partial(permute_pixels, permute_idx=torch.randperm(784))
        permute_dataset = self.load_dataset(permute_transform)

        # plot results -- will need to rewrite this -- 
        # plot_train_results(train_results, test_results, prms)

        results = {} # create a results dictionary

        # return results and trained networks
        return results, nets


    def plot(self, results):
        """
        main plotting loop
        """

        pass # do stuff

    # ----------------------------------------------
    # ------ methods for main experiment loop ------
    # ----------------------------------------------
    def load_networks(self):
        """supporting method for loading networks"""
        nets = [] # see alignment_comparison.py for an example
        return nets
    
    def load_dataset(self, transform_parameters):
        """supporting method for loading the requested dataset"""
        dataset_constructor = get_dataset(self.args.dataset)
        return dataset_constructor(transform_parameters=transform_parameters,
                                   device=self.args.device)

    # ----------------------------------------------
    # ------- methods for main plotting loop -------
    # ----------------------------------------------



        
import torch
from .alignment_stats import AlignmentStatistics
from ..models.registry import get_model, get_model_parameters

class AlignmentComparison(AlignmentStatistics):
    def get_basename(self):
        """
        define basename for the AlignmentComparison experiment
        """
        return 'alignment_comparison'
    
    def prepare_path(self):
        """
        Define save location for each instance of this experiment type
        """
        return [self.args.comparison, self.args.network, self.args.dataset, self.args.optimizer]
    
    def make_args(self, parser):
        """
        Method for adding experiment specific arguments to the argument parser

        (Overwriting the make_args from AlignmentStatistics)
        """

        # Network & Dataset
        parser.add_argument('--network', type=str, default='MLP') # what base network architecture to use
        parser.add_argument('--dataset', type=str, default='MNIST') # what dataset to use
        parser.add_argument('--optimizer', type=str, default='Adam') # what optimizer to train with
        parser.add_argument('--batch-size', type=int, default=1024) # batch size to pass to DataLoader

        # main experiment parameters
        # -- the "comparison" determines what should be compared by the script --
        # -- depending on selection, something about the networks are varied throughout the experiment --
        parser.add_argument('--comparison', type=str, default='lr') # what comparison to do (see load_networks for options)
        parser.add_argument('--regularizers', type=str, nargs='*', default=['none', 'dropout', 'weight_decay'])
        parser.add_argument('--lrs', type=float, nargs='*', default=[1e-2, 1e-3, 1e-4]) # which learning rates to use
        
        # supporting parameters for some of the "comparisons"
        parser.add_argument('--compare-dropout', type=float, default=0.5) # dropout when doing regularizer comparison
        parser.add_argument('--compare-wd', type=float, default=1e-5) # weight-decay when doing regularizer comparison

        # default parameters
        parser.add_argument('--default-lr', type=float, default=1e-3) # default learning rate
        parser.add_argument('--default-dropout', type=float, default=0) # default dropout rate
        parser.add_argument('--default-wd', type=float, default=0) # default weight decay

        # progressive dropout parameters
        parser.add_argument('--num-drops', type=int, default=9, help='number of dropout fractions for progressive dropout')
        parser.add_argument('--dropout-by-layer', default=False, action='store_true', 
                            help='whether to do progressive dropout by layer or across all layers')
        
        # some metaparameters for the experiment
        parser.add_argument('--epochs', type=int, default=100, help='how many epochs to train the networks on')
        parser.add_argument('--replicates', type=int, default=5, help='how many replicates of networks to train') 
        parser.add_argument('--ignore-flag', default=False, action='store_true', help='if used, will omit flagged layers in analyses')
        
        # checkpointing parameters
        parser.add_argument('--use_prev', default=False, action='store_true', help='if used, will pick up training off previous checkpoint')
        parser.add_argument('--save_ckpts', default=False, action='store_true', help='if used, will save checkpoints of models')

        # return parser
        return parser
    

    # ----------------------------------------------
    # ------ methods for main experiment loop ------
    # ----------------------------------------------
    def load_networks(self):
        """
        method for loading networks

        depending on the experiment parameters (which comparison, which metaparams etc)
        this method will create multiple networks with requested parameters and return
        their optimizers and a params dictionary with the experiment parameters associated
        with each network
        """
        model_constructor = get_model(self.args.network)
        model_parameters = get_model_parameters(self.args.network, self.args.dataset)

        # get optimizer
        if self.args.optimizer == 'Adam':
            optim = torch.optim.Adam
        elif self.args.optimizer == 'SGD':
            optim = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer}) not recognized")

        # compare learning rates
        if self.args.comparison == 'lr':
            lrs = [lr for lr in self.args.lrs for _ in range(self.args.replicates)]
            nets = [model_constructor(dropout=self.args.default_dropout, **model_parameters, ignore_flag=self.args.ignore_flag) for _ in lrs]
            nets = [net.to(self.device) for net in nets]
            optimizers = [optim(net.parameters(), lr=lr, weight_decay=self.args.default_wd)
                        for net, lr in zip(nets, lrs)]
            prms = {
                'lrs': lrs, # the value of the independent variable for each network
                'name': 'lr', # the name of the parameter being varied
                'vals': self.args.lrs, # the list of unique values for the relevant parameter
            }
            return nets, optimizers, prms
                
        # compare training with different regularizers
        elif self.args.comparison == 'regularizer':
            dropout_values = [self.args.compare_dropout * (reg == 'dropout') for reg in self.args.regularizers]
            weight_decay_values = [self.args.compare_wd * (reg == 'weight_decay') for reg in self.args.regularizers]
            dropouts = [do for do in dropout_values for _ in range(self.args.replicates)]
            weight_decays = [wd for wd in weight_decay_values for _ in range(self.args.replicates)]
            nets = [model_constructor(dropout=do, **model_parameters, ignore_flag=self.args.ignore_flag) for do in dropouts]
            nets = [net.to(self.device) for net in nets]
            optimizers = [optim(net.parameters(), lr=self.args.default_lr, weight_decay=wd)
                        for net, wd in zip(nets, weight_decays)]
            prms = {
                'dropouts': dropouts, # dropout values by network
                'weight_decays': weight_decays, # weight decay values by network
                'name': 'regularizer', # name of experiment
                'vals': self.args.regularizers, # name of unique regularizers
            }
            return nets, optimizers, prms

        else:
            raise ValueError(f"Comparison={self.args.comparision} is not recognized")



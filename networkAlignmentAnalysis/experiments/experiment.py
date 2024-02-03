from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import os
from tqdm import tqdm
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

from .. import files
from ..datasets import get_dataset
from ..utils import load_checkpoints
from .. import train
from ..utils import (compute_stats_by_type, 
                     load_checkpoints, 
                     named_transpose,
                     transpose_list,
                     rms)

class Experiment(ABC):
    def __init__(self, args=None) -> None:
        """Experiment constructor"""
        self.basename = self.get_basename() # Register basename of experiment
        self.basepath = files.results_path() / self.basename # Register basepath of experiment
        self.get_args(args=args) # Parse arguments to python program
        self.register_timestamp() # Register timestamp of experiment
        self.device = self.args.device

    def report(self, init=False, args=False, meta_args=False) -> None:
        """Method for programmatically reporting details about experiment"""
        # Report general details about experiment 
        if init:
            print(f"Experiment object details:")
            print(f"basename: {self.basename}")
            print(f"basepath: {self.basepath}")
            print(f"experiment folder: {self.get_exp_path()}")
            print('using device: ', self.device)

            # Report any other relevant details
            if self.args.save_networks and self.args.nosave:
                print("Note: setting nosave to True will overwrite save_networks. Nothing will be saved.")

        # Report experiment parameters
        if args:
            for key, val in vars(self.args).items():
                if key in self.meta_args: continue
                print(f"{key}={val}")

        # Report experiment meta parameters
        if meta_args:
            for key, val in vars(self.args).items():
                if key not in self.meta_args: continue
                print(f"{key}={val}")

    def register_timestamp(self) -> None:
        """
        Method for registering formatted timestamp.
        
        If timestamp not provided, then the current time is formatted and used to identify this particular experiment.
        If the timestamp is provided, then that time is used and should identify a previously run and saved experiment.
        """
        if self.args.timestamp is not None:
            self.timestamp = self.args.timestamp
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.args.use_timestamp:
                self.args.timestamp = self.timestamp

    def get_dir(self, create=True) -> Path:
        """
        Method for return directory of target file using prepare_path.
        """
        # Make full path to experiment directory
        exp_path = self.basepath / self.get_exp_path()

        # Make experiment directory if it doesn't yet exist
        if create and not(exp_path.exists()): 
            exp_path.mkdir(parents=True)

        return exp_path
    
    def get_exp_path(self) -> Path:
        """Method for returning child directories of this experiment"""
        # exp_path is the base path followed by whatever folders define this particular experiment
        # (usually things like ['network_name', 'dataset_name', 'test', 'etc'])
        exp_path = Path('/'.join(self.prepare_path()))
        
        # if requested, will also use a timestamp to distinguish this run from others
        if self.args.use_timestamp:
            exp_path = exp_path / self.timestamp

        return exp_path

    
    def get_path(self, name, create=True) -> Path:
        """Method for returning path to file"""
        # get experiment directory
        exp_path = self.get_dir(create=create)

        # return full path (including stem)
        return exp_path / name
    
    @abstractmethod
    def get_basename(self) -> str:
        """Required method for defining the base name of the Experiment"""
        pass

    @abstractmethod
    def prepare_path(self) -> List[str]:
        """
        Required method for defining a pathname for each experiment.

        Must return a list of strings that will be appended to the base path to make an experiment directory.
        See ``get_dir()`` for details.
        """
        pass

    def get_args(self, args=None):
        """
        Method for defining and parsing arguments.
        
        This method defines the standard arguments used for any Experiment, and
        the required method make_args() is used to add any additional arguments
        specific to each experiment.
        """
        self.meta_args = [] # a list of arguments that shouldn't be updated when loading an old experiment
        parser = ArgumentParser(description=f"arguments for {self.basename}")
        parser = self.make_args(parser)
        
        # saving and new experiment loading parameters
        parser.add_argument('--nosave', default=False, action='store_true', help="prevents saving of results or plots")
        parser.add_argument('--justplot', default=False, action='store_true', help="plot saved data without retraining and analyzing networks")
        parser.add_argument('--save-networks', default=False, action='store_true', help="if --nosave wasn't provided, will also save networks that are trained")
        parser.add_argument('--showprms', default=False, action='store_true', help='show parameters of previously saved experiment without doing anything else')
        parser.add_argument('--showall', default=False, action='store_true', help='if true, will show all plots at once rather than having the user close each one for the next')
        parser.add_argument('--device', type=str, default=None, help='which device to use (automatic if not provided)')

        # add meta arguments 
        self.meta_args += ['nosave', 'justplot', 'save_networks', 'showprms', 'showall', 'device']
        
        # common parameters that shouldn't be updated when loading old experiment
        parser.add_argument('--use-timestamp', default=False, action='store_true', help='if used, will save data in a folder named after the current time (or whatever is provided in --timestamp)')
        parser.add_argument('--timestamp', default=None, help='the timestamp of a previous experiment to plot or observe parameters')
        
        # parse arguments (passing directly because initial parser will remove the "--experiment" argument)
        self.args = parser.parse_args(args=args)

        # manage device
        if self.args.device is None:
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # do checks
        if self.args.use_timestamp and self.args.justplot:
            assert self.args.timestamp is not None, "if use_timestamp=True and plotting stored results, must provide a timestamp"
    
    @abstractmethod
    def make_args(self, parser) -> ArgumentParser:
        """
        Required method for defining special-case arguments.

        This should just use the add_argument method on the parser provided as input.
        """
        pass

    def get_prms_path(self):
        """Method for loading path to experiment parameters file"""
        return self.get_dir() / 'prms.pth'
    
    def get_results_path(self):
        """Method for loading path to experiment results files"""
        return self.get_dir() / 'results.pth'
    
    def get_network_path(self, name):
        """Method for loading path to saved network file"""
        return self.get_dir() / f"{name}.pt"
    
    def get_checkpoint_path(self):
        '''Method for loading path to network checkpoint file'''
        return self.get_dir() / 'checkpoint.tar'

    def _update_args(self, prms):
        """Method for updating arguments from saved parameter dictionary"""
        # First check if saved parameters contain unknown keys
        if prms.keys() > vars(self.args).keys():
            raise ValueError(f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(self.args).keys())}")
        
        # Then update self.args while ignoring any meta arguments
        for ak in vars(self.args):
            if ak in self.meta_args: continue # don't update meta arguments
            if ak in prms and prms[ak] != vars(self.args)[ak]:
                print(f"Requested argument {ak}={vars(self.args)[ak]} differs from saved, which is: {ak}={prms[ak]}. Using saved...")
                setattr(self.args, ak, prms[ak])

    def save_experiment(self, results):
        """Method for saving experiment parameters and results to file"""
        # Save experiment parameters
        torch.save(vars(self.args), self.get_prms_path())
        # Save experiment results 
        torch.save(results, self.get_results_path())

    def load_experiment(self, no_results=False):
        """Method for loading saved experiment parameters and results"""
        # Check if prms path is there
        if not self.get_prms_path().exists():
            raise ValueError(f"saved parameters at: f{self.get_prms_path()} not found!")
        
        # Check if results directory is there
        if not self.get_results_path().exists():
            raise ValueError(f"saved results at: f{self.get_results_path()} not found!")

        # Load parameters into object
        prms = torch.load(self.get_prms_path())
        self._update_args(prms)
        
        # Don't load results if requested
        if no_results: return None

        # Load and return results
        return torch.load(self.get_results_path())
    
    def save_networks(self, nets, id=None):
        """
        Method for saving any networks that were trained
        
        Names networks with index in list of **nets**
        If **id** is provided, will use id in addition to the index
        """
        name = f"net_{id}_" if id is not None else "net_"
        for idx, net in enumerate(nets):
            cname = name + f"{idx}"
            torch.save(net, self.get_network_path(cname))

    @abstractmethod
    def main(self) -> Tuple[Dict, List[torch.nn.Module]]:
        """
        Required method for operating main experiment functions. 
        
        This method should perform any core training and analyses related to the experiment
        and return a results dictionary and a list of pytorch nn.Modules. The second requirement
        (torch modules) can probably be relaxed, but doesn't need to yet so let's keep it as is
        for overall clarity.
        """
        pass

    @abstractmethod
    def plot(self, results: Dict) -> None:
        """
        Required method for operating main plotting functions.
        
        Should accept as input a results dictionary and run plotting functions.
        If any plots are to be saved, then each plotting function must do so 
        accordingly. 
        """
        pass

    # -- support for main processing loop --
    def prepare_dataset(self, transform_parameters):
        """simple method for getting dataset """    
        return get_dataset(self.args.dataset,
                           build=True,
                           transform_parameters=transform_parameters,
                           loader_parameters={'batch_size': self.args.batch_size},
                           device=self.args.device)
    
    def train_networks(self, nets, optimizers, dataset):
        """train and test networks"""
        # do training loop
        parameters = dict(
            train_set=True,
            num_epochs=self.args.epochs,
            alignment=not(self.args.no_alignment),
            delta_weights=self.args.delta_weights,
            frequency=self.args.frequency,
        )

        if self.args.use_prev & os.path.isfile(self.get_checkpoint_path()):
            nets, optimizers, results = load_checkpoints(nets,
                                                         optimizers,
                                                         self.args.device,
                                                         self.get_checkpoint_path())
            for net in nets:
                net.train()

            parameters['num_complete'] = results['epoch'] + 1
            parameters['results'] = results
            print('loaded networks from previous checkpoint')

        if self.args.save_ckpts:
            parameters['save_checkpoints'] = (True, 1, self.get_checkpoint_path(), self.args.device)

        print('training networks...')
        train_results = train.train(nets, optimizers, dataset, **parameters)

        # do testing loop
        print('testing networks...')
        parameters['train_set'] = False
        test_results = train.test(nets, dataset, **parameters)

        return train_results, test_results
    
    def progressive_dropout_experiment(self, nets, dataset, alignment=None, train_set=False):
        """
        perform a progressive dropout (of nodes) experiment
        alignment is optional, but will be recomputed if you've already measured it. You can provide it
        by setting: alignment=test_results['alignment'] if ``train_networks`` has already been run.
        """
        # do targeted dropout experiment
        print('performing targeted dropout...')
        dropout_parameters = dict(num_drops=self.args.num_drops, by_layer=self.args.dropout_by_layer, train_set=train_set)
        dropout_results = train.progressive_dropout(nets, dataset, alignment=alignment, **dropout_parameters)
        return dropout_results, dropout_parameters
    
    def measure_eigenfeatures(self, nets, dataset, train_set=False):
        # measure eigenfeatures
        print('measuring eigenfeatures...')
        beta, eigvals, eigvecs, class_betas = [], [], [], []
        dataloader = dataset.train_loader if train_set else dataset.test_loader
        for net in tqdm(nets):
            eigenfeatures = net.measure_eigenfeatures(dataloader, with_updates=False)
            beta_by_class = net.measure_class_eigenfeatures(dataloader, eigenfeatures[2], rms=False, with_updates=False)
            beta.append(eigenfeatures[0])
            eigvals.append(eigenfeatures[1])
            eigvecs.append(eigenfeatures[2])
            class_betas.append(beta_by_class)

        # make it a dictionary
        return dict(beta=beta, eigvals=eigvals, eigvecs=eigvecs, class_betas=class_betas, class_names=dataloader.dataset.classes) 
        
    def eigenvector_dropout(self, nets, dataset, eigen_results, train_set=False):
        """
        do targeted eigenvector dropout with precomputed eigenfeatures
        """
        # do targeted dropout experiment
        print('performing targeted eigenvector dropout...')
        evec_dropout_parameters = dict(num_drops=self.args.num_drops, by_layer=self.args.dropout_by_layer, train_set=train_set)
        evec_dropout_results = train.eigenvector_dropout(nets, dataset, eigen_results['eigvals'], eigen_results['eigvecs'], **evec_dropout_parameters)
        return evec_dropout_results, evec_dropout_parameters
    

    def plot_ready(self, name):
        """standard method for saving and showing plot when it's ready"""
        # if saving, then save the plot
        if not self.args.nosave:
            plt.savefig(str(self.get_path(name)))
        # show the plot now if not doing showall
        if not self.args.showall:
            plt.show()


    # ----------------------------------------------
    # ------- methods for main plotting loop -------
    # ----------------------------------------------
    def plot_train_results(self, train_results, test_results, prms):
        """
        plotting method for training trajectories and testing data
        """

        num_train_epochs = train_results['loss'].size(0)
        num_types = len(prms['vals'])
        labels = [f"{prms['name']}={val}" for val in prms['vals']]

        print("getting statistics on run data...")
        alignment = torch.stack([torch.mean(align, dim=2) for align in train_results['alignment']])
        
        cmap = mpl.colormaps['tab10']

        train_loss_mean, train_loss_se = compute_stats_by_type(train_results['loss'], 
                                                                num_types=num_types, dim=1, method='se')
        train_acc_mean, train_acc_se = compute_stats_by_type(train_results['accuracy'],
                                                                num_types=num_types, dim=1, method='se')

        align_mean, align_se = compute_stats_by_type(alignment, num_types=num_types, dim=1, method='se')

        test_loss_mean, test_loss_se = compute_stats_by_type(torch.tensor(test_results['loss']),
                                                                num_types=num_types, dim=0, method='se')
        test_acc_mean, test_acc_se = compute_stats_by_type(torch.tensor(test_results['accuracy']),
                                                                num_types=num_types, dim=0, method='se')


        print("plotting run data...")
        xOffset = [-0.2, 0.2]
        get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]

        # Make Training and Testing Performance Figure
        alpha = 0.3
        figdim = 3
        figratio = 2
        width_ratios = [figdim, figdim/figratio, figdim, figdim/figratio]

        fig, ax = plt.subplots(1, 4, figsize=(sum(width_ratios), figdim), width_ratios=width_ratios, layout='constrained')

        # plot loss results fot training and testing
        for idx, label in enumerate(labels):
            cmn = train_loss_mean[:, idx]
            cse = train_loss_se[:, idx]
            tmn = test_loss_mean[idx]
            tse = test_loss_se[idx]

            ax[0].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
            ax[0].fill_between(range(num_train_epochs), cmn+cse, cmn-cse, color=(cmap(idx), alpha))
            ax[1].plot(get_x(idx), [tmn]*2, color=cmap(idx), label=label, lw=4)
            ax[1].plot([idx, idx], [tmn-tse, tmn+tse], color=cmap(idx), lw=1.5)
            
        ax[0].set_xlabel('Training Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')
        ax[0].set_ylim(0, None)
        ylims = ax[0].get_ylim()
        ax[1].set_xticks(range(num_types))
        ax[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Testing')
        ax[1].set_xlim(-0.5, num_types-0.5)
        ax[1].set_ylim(ylims)

        # plot loss results fot training and testing
        for idx, label in enumerate(labels):
            cmn = train_acc_mean[:, idx]
            cse = train_acc_se[:, idx]
            tmn = test_acc_mean[idx]
            tse = test_acc_se[idx]

            ax[2].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
            ax[2].fill_between(range(num_train_epochs), cmn+cse, cmn-cse, color=(cmap(idx), alpha))
            ax[3].plot(get_x(idx), [tmn]*2, color=cmap(idx), label=label, lw=4)
            ax[3].plot([idx, idx], [tmn-tse, tmn+tse], color=cmap(idx), lw=1.5)
            
        ax[2].set_xlabel('Training Epoch')
        ax[2].set_ylabel('Accuracy (%)')
        ax[2].set_title('Training Accuracy')
        ax[2].set_ylim(0, 100)
        ax[3].set_xticks(range(num_types))
        ax[3].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax[3].set_ylabel('Accuracy (%)')
        ax[3].set_title('Testing')
        ax[3].set_xlim(-0.5, num_types-0.5)
        ax[3].set_ylim(0, 100)

        self.plot_ready('train_test_performance')

        # Make Alignment Figure
        num_align_epochs = align_mean.size(2)
        num_layers = align_mean.size(0)
        fig, ax = plt.subplots(1, num_layers, figsize=(num_layers*figdim, figdim), layout='constrained', sharex=True)
        for idx, label in enumerate(labels):
            for layer in range(num_layers):
                cmn = align_mean[layer, idx] * 100
                cse = align_se[layer, idx] * 100
                ax[layer].plot(range(num_align_epochs), cmn, color=cmap(idx), label=label)
                ax[layer].fill_between(range(num_align_epochs), cmn+cse, cmn-cse, color=(cmap(idx), alpha))

        for layer in range(num_layers):
            ax[layer].set_ylim(0, None)
            ax[layer].set_xlabel('Training Epoch')
            ax[layer].set_ylabel('Alignment (%)')
            ax[layer].set_title(f"Layer {layer}")

        ax[0].legend(loc='lower right')

        self.plot_ready('train_alignment_by_layer')


    def plot_dropout_results(self, dropout_results, dropout_parameters, prms, dropout_type='nodes'):
        num_types = len(prms['vals'])
        labels = [f"{prms['name']}={val} - dropout {dropout_type}" for val in prms['vals']]
        cmap = mpl.colormaps['Set1']
        alpha = 0.3
        msize = 10
        figdim = 3

        num_layers = dropout_results['progdrop_loss_high'].size(2)
        names = ['From high', 'From low', 'Random']
        num_exp = len(names)
        dropout_fraction = dropout_results['dropout_fraction']
        by_layer = dropout_results['by_layer']
        extra_name = 'by_layer' if by_layer else 'all_layers'
        extra_name += dropout_type

        # Get statistics across each network type for progressive dropout experiment
        print("measuring statistics on dropout analysis...")
        loss_mean_high, loss_se_high = compute_stats_by_type(dropout_results['progdrop_loss_high'], 
                                                                num_types=num_types, dim=0, method='se')
        loss_mean_low, loss_se_low = compute_stats_by_type(dropout_results['progdrop_loss_low'], 
                                                                num_types=num_types, dim=0, method='se')
        loss_mean_rand, loss_se_rand = compute_stats_by_type(dropout_results['progdrop_loss_rand'], 
                                                                num_types=num_types, dim=0, method='se')

        acc_mean_high, acc_se_high = compute_stats_by_type(dropout_results['progdrop_acc_high'], 
                                                                num_types=num_types, dim=0, method='se')
        acc_mean_low, acc_se_low = compute_stats_by_type(dropout_results['progdrop_acc_low'], 
                                                                num_types=num_types, dim=0, method='se')
        acc_mean_rand, acc_se_rand = compute_stats_by_type(dropout_results['progdrop_acc_rand'], 
                                                                num_types=num_types, dim=0, method='se')

        # Contract into lists for looping through to plot
        loss_mean = [loss_mean_high, loss_mean_low, loss_mean_rand]
        loss_se = [loss_se_high, loss_se_low, loss_se_rand]
        acc_mean = [acc_mean_high, acc_mean_low, acc_mean_rand]
        acc_se = [acc_se_high, acc_se_low, acc_se_rand]


        print("plotting dropout results...")
        # Plot Loss for progressive dropout experiment
        fig, ax = plt.subplots(num_layers, num_types, figsize=(num_types*figdim, num_layers*figdim), sharex=True, sharey=True, layout='constrained')
        ax = np.reshape(ax, (num_layers, num_types))

        for idx, label in enumerate(labels):
            for layer in range(num_layers):
                for iexp, name in enumerate(names):
                    cmn = loss_mean[iexp][idx, :, layer]
                    cse = loss_se[iexp][idx, :, layer]
                    ax[layer, idx].plot(dropout_fraction, cmn, color=cmap(iexp), marker='.', markersize=msize, label=name)
                    ax[layer, idx].fill_between(dropout_fraction, cmn+cse, cmn-cse, color=(cmap(iexp), alpha))
            
                if layer==0:
                    ax[layer, idx].set_title(label)

                if layer==num_layers-1:
                    ax[layer, idx].set_xlabel('Dropout Fraction')
                    ax[layer, idx].set_xlim(0, 1)
                
                if idx==0:
                    ax[layer, idx].set_ylabel('Loss w/ Dropout')

                if iexp==num_exp-1:
                    ax[layer, idx].legend(loc='best')
        
        self.plot_ready('prog_dropout_'+extra_name+'_loss')


        fig, ax = plt.subplots(num_layers, num_types, figsize=(num_types*figdim, num_layers*figdim), sharex=True, sharey=True, layout='constrained')
        ax = np.reshape(ax, (num_layers, num_types))

        for idx, label in enumerate(labels):
            for layer in range(num_layers):
                for iexp, name in enumerate(names):
                    cmn = acc_mean[iexp][idx, :, layer]
                    cse = acc_se[iexp][idx, :, layer]
                    ax[layer, idx].plot(dropout_fraction, cmn, color=cmap(iexp), marker='.', markersize=msize, label=name)
                    ax[layer, idx].fill_between(dropout_fraction, cmn+cse, cmn-cse, color=(cmap(iexp), alpha))

                ax[layer, idx].set_ylim(0, 100)

                if layer==0:
                    ax[layer, idx].set_title(label)

                if layer==num_layers-1:
                    ax[layer, idx].set_xlabel('Dropout Fraction')
                    ax[layer, idx].set_xlim(0, 1)
                
                if idx==0:
                    ax[layer, idx].set_ylabel('Accuracy w/ Dropout')

                if iexp==num_exp-1:
                    ax[layer, idx].legend(loc='best')
        
        self.plot_ready('prog_dropout_'+extra_name+'_accuracy')


    def plot_eigenfeatures(self, results, prms):
        """method for plotting results related to eigen-analysis"""
        beta, eigvals, class_betas, class_names = results['beta'], results['eigvals'], results['class_betas'], results['class_names']
        beta = [[torch.abs(b) for b in net_beta] for net_beta in beta]
        class_betas = [[rms(cb, dim=2) for cb in net_class_beta] for net_class_beta in class_betas]

        num_types = len(prms['vals'])
        labels = [f"{prms['name']}={val}" for val in prms['vals']]
        cmap = mpl.colormaps['tab10']
        class_cmap = mpl.colormaps['viridis'].resampled(len(class_names))

        print("measuring statistics of eigenfeature analyses...")

        # shape wrangling
        beta = [torch.stack(b) for b in transpose_list(beta)]
        eigvals = [torch.stack(ev) for ev in transpose_list(eigvals)]
        class_betas = [torch.stack(cb) for cb in transpose_list(class_betas)]

        # normalize to relative values
        beta = [b / b.sum(dim=2, keepdim=True) for b in beta]
        eigvals = [ev / ev.sum(dim=1, keepdim=True) for ev in eigvals]
        class_betas = [cb / cb.sum(dim=2, keepdim=True) for cb in class_betas]

        # reuse these a few times
        statprms = lambda method: dict(num_types=num_types, dim=0, method=method)

        # get mean and variance eigenvalues for each layer for each network type
        mean_evals, var_evals = named_transpose([compute_stats_by_type(ev, **statprms('var')) for ev in eigvals])

        # get sorted betas (sorted within each neuron)
        sorted_beta = [torch.sort(b, descending=True, dim=2).values for b in beta]

        # get mean / se beta for each layer for each network type
        mean_beta, se_beta = named_transpose([compute_stats_by_type(b, **statprms('var')) for b in beta])
        mean_sorted, se_sorted = named_transpose([compute_stats_by_type(b, **statprms('var')) for b in sorted_beta])
        mean_class_beta, se_class_beta = named_transpose([compute_stats_by_type(cb, **statprms('var')) for cb in class_betas])

        print("plotting eigenfeature results...")
        figdim = 3
        alpha = 0.3
        num_layers = len(mean_beta)
        fig, ax = plt.subplots(2, num_layers, figsize=(num_layers*figdim, figdim*2), layout='constrained')

        for layer in range(num_layers):
            num_input = mean_evals[layer].size(1)
            num_nodes = mean_beta[layer].size(1)
            for idx, label in enumerate(labels):
                mn_ev = mean_evals[layer][idx]
                se_ev = var_evals[layer][idx]
                mn_beta = torch.mean(mean_beta[layer][idx], dim=0)
                se_beta = torch.std(mean_beta[layer][idx], dim=0) / np.sqrt(num_nodes)
                mn_sort = torch.mean(mean_sorted[layer][idx], dim=0)
                se_sort = torch.std(mean_sorted[layer][idx], dim=0) / np.sqrt(num_nodes)
                ax[0, layer].plot(range(num_input), mn_ev, color=cmap(idx), linestyle='--', label='eigvals' if idx==0 else None)
                ax[0, layer].plot(range(num_input), mn_beta, color=cmap(idx), label=label)
                ax[0, layer].fill_between(range(num_input), mn_beta+se_beta, mn_beta-se_beta, color=(cmap(idx), alpha))
                ax[1, layer].plot(range(num_input), mn_sort, color=cmap(idx), label=label)
                ax[1, layer].fill_between(range(num_input), mn_sort+se_sort, mn_sort-se_sort, color=(cmap(idx), alpha))
                
                ax[0, layer].set_xscale('log')
                ax[1, layer].set_xscale('log')
                ax[0, layer].set_xlabel('Input Dimension')
                ax[1, layer].set_xlabel('Sorted Input Dim')
                ax[0, layer].set_ylabel('Relative Eigval & Beta')
                ax[1, layer].set_ylabel('Relative Beta (Sorted)')
                ax[0, layer].set_title(f"Layer {layer}")
                ax[1, layer].set_title(f"Layer {layer}")

                if layer==num_layers-1:
                    ax[0, layer].legend(loc='best')
                    ax[1, layer].legend(loc='best')

        self.plot_ready('eigenfeatures')


        fig, ax = plt.subplots(1, num_layers, figsize=(num_layers*figdim, figdim), layout='constrained')

        for layer in range(num_layers):
            num_input = mean_evals[layer].size(1)
            num_nodes = mean_beta[layer].size(1)
            for idx, label in enumerate(labels):
                mn_ev = mean_evals[layer][idx]
                se_ev = var_evals[layer][idx]
                mn_beta = torch.mean(mean_beta[layer][idx], dim=0)
                se_beta = torch.std(mean_beta[layer][idx], dim=0) / np.sqrt(num_nodes)
                mn_sort = torch.mean(mean_sorted[layer][idx], dim=0)
                se_sort = torch.std(mean_sorted[layer][idx], dim=0) / np.sqrt(num_nodes)
                ax[layer].plot(range(num_input), mn_ev, color=cmap(idx), linestyle='--', label='eigvals' if idx==0 else None)
                ax[layer].plot(range(num_input), mn_beta, color=cmap(idx), label=label)
                ax[layer].fill_between(range(num_input), mn_beta+se_beta, mn_beta-se_beta, color=(cmap(idx), alpha))

                ax[layer].set_xscale('log')
                ax[layer].set_yscale('log')
                ax[layer].set_xlabel('Input Dimension')
                ax[layer].set_ylabel('Relative Eigval & Beta')
                ax[layer].set_title(f"Layer {layer}")

                if layer==num_layers-1:
                    ax[layer].legend(loc='best')

        self.plot_ready('eigenfeatures_loglog')


        fig, ax = plt.subplots(num_types, num_layers, figsize=(num_layers*figdim, figdim*num_types), layout='constrained')
        ax = np.reshape(ax, (num_types, num_layers))
        for layer in range(num_layers):
            num_input = mean_evals[layer].size(1)
            for idx, label in enumerate(labels):                
                for idx_class, class_name in enumerate(class_names):
                    mn_data = mean_class_beta[layer][idx][idx_class]
                    se_data = se_class_beta[layer][idx][idx_class]
                    ax[idx, layer].plot(range(num_input), mn_data, color=class_cmap(idx_class), label=class_name)
                    ax[idx, layer].fill_between(range(num_input), mn_data+se_data, mn_data-se_data, color=(class_cmap(idx_class), alpha))
                    
                ax[idx, layer].set_xscale('log')
                ax[idx, layer].set_yscale('linear')
                ax[idx, layer].set_xlabel('Input Dimension')
                if layer==0:
                    ax[idx, layer].set_ylabel(f"{label}\nClass Loading (RMS)")
                if idx==0:
                    ax[idx, layer].set_title(f"Layer {layer}")
                
                if layer==num_layers-1:
                    ax[idx, layer].legend(loc='upper right', fontsize=6)

        self.plot_ready('class_eigenfeatures')

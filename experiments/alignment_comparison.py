import sys
import os

# add path that contains the alignment package
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datetime import datetime
import numpy as np
import torch
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import matplotlib as mpl

from networkAlignmentAnalysis import files
from networkAlignmentAnalysis.models.registry import get_model
from networkAlignmentAnalysis.datasets import get_dataset
from networkAlignmentAnalysis import train
from networkAlignmentAnalysis.utils import avg_from_full, compute_stats_by_type

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_NAME = 'alignment_comparison'
NETWORK_PATH = files.results_path() / BASE_NAME
RESULTS_PATH = files.results_path() / BASE_NAME

# register the timestamp of the run everytime this file is executed
def register_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_path(args, name, network=False):
    """Method for retrieving paths for saving networks or data"""
    base_path = NETWORK_PATH if network else RESULTS_PATH
    exp_path = base_path / args.comparison # path to this experiment 

    # use timestamp to save each run independently (or not to have a "master" run)
    if args.use_timestamp:
        exp_path = exp_path / run_timestamp 
    
    # Make experiment directory if it doesn't yet exist
    if not exp_path.exists(): 
        exp_path.mkdir()

    # return full path
    return exp_path / name


def get_args():
    """Contained method for defining and parsing arguments for programmatic runs"""
    parser = ArgumentParser(description='alignment_comparison')
    
    parser.add_argument('--network', type=str, default='MLP') # what base network architecture to use
    parser.add_argument('--dataset', type=str, default='MNIST') # what dataset to use

    # main experiment parameters
    parser.add_argument('--comparison', type=str, default='lr') # what comparison to do
    parser.add_argument('--lrs', type=float, nargs='*', default=[1e-3, 5e-4, 1e-4]) # which learning rates to use

    # progressive dropout parameters
    parser.add_argument('--num_drops', type=int, default=9, help='number of dropout fractions for progressive dropout')
    parser.add_argument('--dropout_by_layer', default=False, action='store_true', 
                        help='whether to do progressive dropout by layer or across all layers')
    
    # some metaparameters for the experiment
    parser.add_argument('--epochs', type=int, default=1) # how many rounds of training to do
    parser.add_argument('--replicates', type=int, default=1) # how many copies of identical networks to train
    
    # saving parameters
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--use-timestamp', default=False, action='store_true')
    parser.add_argument('--save-networks', default=False, action='store_true')
    
    # any additional argument checks go here:
    
    # return parsed arguments
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
    train_results = train.train(nets, optimizers, dataset, **parameters)
    test_results = train.test(nets, dataset, **parameters)

    return train_results, test_results, prms


def plot_train_results(train_results, test_results, prms):
    num_train_epochs = train_results['loss'].size(0)
    num_types = len(prms['vals'])
    labels = [f"{prms['name']}={val}" for val in prms['vals']]
    alignment = torch.stack([avg_from_full(align).T for align in train_results['alignment']])
    cmap = mpl.colormaps['tab10']

    train_loss_mean, train_loss_se = compute_stats_by_type(train_results['loss'], 
                                                            num_types=num_types, dim=1, method='se')
    train_acc_mean, train_acc_se = compute_stats_by_type(train_results['accuracy'],
                                                            num_types=num_types, dim=1, method='se')
    align_mean, align_se = compute_stats_by_type(alignment, num_types=num_types, dim=0, method='se')

    test_loss_mean, test_loss_se = compute_stats_by_type(torch.tensor(test_results['loss']),
                                                            num_types=num_types, dim=0, method='se')
    test_acc_mean, test_acc_se = compute_stats_by_type(torch.tensor(test_results['accuracy']),
                                                            num_types=num_types, dim=0, method='se')

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

    if not args.nosave:
        plt.savefig(str(get_path(args, 'train_test_performance')))

    plt.show()


    # Make Alignment Figure
    num_layers = align_mean.size(2)
    fig, ax = plt.subplots(1, num_layers, figsize=(num_layers*figdim, figdim), layout='constrained', sharex=True)
    for idx, label in enumerate(labels):
        for layer in range(num_layers):
            cmn = align_mean[idx, :, layer] * 100
            cse = align_se[idx, :, layer] * 100
            ax[layer].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
            ax[layer].fill_between(range(num_train_epochs), cmn+cse, cmn-cse, color=(cmap(idx), alpha))

    for layer in range(num_layers):
        ax[layer].set_ylim(0, None)
        ax[layer].set_xlabel('Training Epoch')
        ax[layer].set_ylabel('Alignment (%)')
        ax[layer].set_title(f"Layer {layer}")

    ax[0].legend(loc='lower right')

    if not args.nosave:
        plt.savefig(str(get_path(args, 'train_alignment_by_layer')))

    plt.show()


def plot_dropout_results(dropout_results, prms, dropout_parameters):
    num_types = len(prms['vals'])
    labels = [f"{prms['name']}={val}" for val in prms['vals']]
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

    # Get statistics across each network type for progressive dropout experiment
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


    # Plot Loss for progressive dropout experiment
    fig, ax = plt.subplots(num_layers, num_types, figsize=(num_exp*figdim, num_types*figdim), sharex=True, sharey=True, layout='constrained')
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
            
            if iexp==0:
                ax[layer, idx].set_ylabel('Loss w/ Dropout')

            if iexp==num_exp-1:
                ax[layer, idx].legend(loc='best')
        
    if not args.nosave:
        plt.savefig(str(get_path(args, 'prog_dropout_'+extra_name+'_loss')))

    plt.show()


    fig, ax = plt.subplots(num_layers, num_types, figsize=(num_layers*figdim, num_types*figdim), sharex=True, sharey=True, layout='constrained')
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
            
            if iexp==0:
                ax[layer, idx].set_ylabel('Accuracy w/ Dropout')

            if iexp==num_exp-1:
                ax[layer, idx].legend(loc='best')
        
    if not args.nosave:
        plt.savefig(str(get_path(args, 'prog_dropout_'+extra_name+'_accuracy')))

    plt.show()


if __name__ == '__main__':

    # run parameters
    args = get_args()

    # get timestamp
    run_timestamp = register_timestamp()

     # get networks
    nets, optimizers, prms = load_networks(args)

    # load dataset
    dataset = load_dataset(args, nets[0].get_transform_parameters(args.dataset))

    # do training 
    train_results, test_results, prms = train_networks(nets, optimizers, dataset, args)

    # do targeted dropout experiment
    dropout_parameters = dict(num_drops=args.num_drops, by_layer=args.dropout_by_layer)
    dropout_results = train.progressive_dropout(nets, dataset, **dropout_parameters)

    # measure eigenfeatures


    # plot results
    plot_train_results(train_results, test_results, prms)
    plot_dropout_results(dropout_results, prms, dropout_parameters)
    



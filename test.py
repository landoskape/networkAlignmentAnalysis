import time
from tqdm import tqdm
import torch
from torch import nn

from networkAlignmentAnalysis.models.registry import get_model
from networkAlignmentAnalysis.datasets import get_dataset
from networkAlignmentAnalysis import train

from argparse import ArgumentParser

def get_args(args=None):
    parser = ArgumentParser(description='test alignment code')
    parser.add_argument('--network', type=str, default='MLP')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args(args=args)

if __name__ == '__main__':
    args = get_args()

    DEVICE = args.device if args.device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device: ', DEVICE)

    # get network
    net_args = dict(
        ignore_flag=False,
    )

    net = get_model(args.network, build=True, **net_args).to(DEVICE)
    dataset = get_dataset(args.dataset, build=True, transform_parameters=net, device=DEVICE)

    # optim = torch.optim.Adam(net.parameters(), lr=1e-2)

    # results = train.train([net], [optim], dataset, train_set=True, num_epochs=50, alignment=False)

    betas, eigenvalues, eigenvectors = net.measure_eigenfeatures(dataset.test_loader)
    net.shape_eigenfeatures(net.get_alignment_layer_indices(), eigenvalues, eigenvectors, lambda x: x)


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
    return parser.parse_args(args=args)

if __name__ == '__main__':
    args = get_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device: ', DEVICE)

    # get network
    net = get_model(args.network, build=True).to(DEVICE)
    dataset = get_dataset(args.dataset, build=True, transform_parameters=net)

    betas, eigenvalues, eigenvectors = net.measure_eigenfeatures(dataset.test_loader)

    beta_by_class = net.measure_class_eigenfeatures(dataset.test_loader, eigenvectors)
    print('no rms:', [b.shape for b in beta_by_class])

    beta_by_class = net.measure_class_eigenfeatures(dataset.test_loader, eigenvectors, rms=True)
    print('rms=True:', [b.shape for b in beta_by_class])
    

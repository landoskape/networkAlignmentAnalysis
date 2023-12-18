import time
from tqdm import tqdm
import torch
from torch import nn

from networkAlignmentAnalysis.models.models import MLP, CNN2P2, AlexNet
from networkAlignmentAnalysis import datasets
from networkAlignmentAnalysis import train

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='test alignment code')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--epochs', type=int, default=1)
    return parser.parse_args()

networks = {
    'MLP': MLP,
    'CNN2P2': CNN2P2,
    'AlexNet': AlexNet,
}

dataset_dict = {
    'MNIST': datasets.MNIST,
}

if __name__ == '__main__':
    args = get_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device: ', DEVICE)

    # get network(s)
    nets = [MLP(), MLP()]
    nets = [net.to(DEVICE) for net in nets]

    # create optimizer
    optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in nets]

    # load dataset
    dataset = dataset_dict.get(args.dataset)(transform_parameters=nets[0].get_transform_parameters(args.dataset))

    # do training loop
    parameters = dict(
        train_set=True,
        num_epochs=args.epochs,
        alignment=True,
        delta_weights=True,
    )
    results = train.train(nets, optimizers, dataset, **parameters)


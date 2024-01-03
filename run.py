import time
from tqdm import tqdm
import torch
from torch import nn

from networkAlignmentAnalysis.models.registry import get_model
from networkAlignmentAnalysis.datasets import get_dataset
from networkAlignmentAnalysis import train

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='test alignment code')
    parser.add_argument('--network', type=str, default='MLP')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--replicates', type=int, default=2)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device: ', DEVICE)

    # get network(s)
    nets = [get_model(args.network, build=True) for _ in range(args.replicates)]
    nets = [net.to(DEVICE) for net in nets]

    # create optimizer
    optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in nets]

    # load dataset
    dataset = get_dataset(args.dataset, build=True, transform_parameters=nets[0].get_transform_parameters(args.dataset))

    # # do training loop
    # parameters = dict(
    #     train_set=True,
    #     num_epochs=args.epochs,
    #     alignment=True,
    #     delta_weights=True,
    # )
    # results = train.train(nets, optimizers, dataset, **parameters)

    outputs = nets[0].measure_eigenfeatures(dataset.test_loader)


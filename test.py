import time
from tqdm import tqdm
import torch
from torch import nn

from networkAlignmentAnalysis.models.registry import get_model
from networkAlignmentAnalysis.datasets import get_dataset
from networkAlignmentAnalysis import train

from argparse import ArgumentParser


def get_args(args=None):
    parser = ArgumentParser(description="test alignment code")
    parser.add_argument("--network", type=str, default="CNN2P2")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-alignment", default=False, action="store_true")
    parser.add_argument(
        "--ignore-flag",
        default=False,
        action="store_true",
        help="if used, will omit flagged layers in analyses",
    )
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = get_args()

    DEVICE = (
        args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("using device:", DEVICE, "DeviceCount:", torch.cuda.device_count())

    # get network
    net_args = dict(ignore_flag=args.ignore_flag)

    net = get_model(args.network, build=True, **net_args).to(DEVICE)

    loader_parameters = dict(
        batch_size=1024,
    )
    dataset = get_dataset(
        args.dataset,
        build=True,
        transform_parameters=net,
        loader_parameters=loader_parameters,
        dataset_parameters=dict(download=True),
        device=DEVICE,
    )

    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    alignment = False if args.no_alignment else True
    results = train.train([net], [optim], dataset, train_set=False, num_epochs=1, alignment=True)

    inputs, labels = net._process_collect_activity(
        dataset, train_set=False, with_updates=True, use_training_mode=True
    )
    betas, eigenvalues, eigenvectors = net.measure_eigenfeatures(inputs)
    # net.shape_eigenfeatures(net.get_alignment_layer_indices(), eigenvalues, eigenvectors, lambda x: x)

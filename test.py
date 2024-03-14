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
    parser.add_argument("--network", type=str, default="MLP")
    # parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--device", type=str, default=None)
    # parser.add_argument("--no-alignment", default=False, action="store_true")
    # parser.add_argument(
    #     "--ignore-flag",
    #     default=False,
    #     action="store_true",
    #     help="if used, will omit flagged layers in analyses",
    # )
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = get_args()

    DEVICE = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", DEVICE, "DeviceCount:", torch.cuda.device_count())

    model_name = args.network
    dataset_name = "MNIST"

    net = get_model(model_name, build=True, dataset=dataset_name, dropout=0.0, ignore_flag=False)
    net.to(DEVICE)

    loader_parameters = dict(
        shuffle=True,
    )
    dataset = get_dataset(dataset_name, build=True, transform_parameters=net, loader_parameters=loader_parameters, device=DEVICE)

    optimizer = torch.optim.Adam(net.parameters())

    parameters = dict(
        num_epochs=2,
        compare_expected=True,
    )
    results = train.train([net], [optimizer], dataset, **parameters)

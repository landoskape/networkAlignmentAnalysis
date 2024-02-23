from networkAlignmentAnalysis.datasets import get_dataset
from argparse import ArgumentParser

def get_args(args=None):
    parser = ArgumentParser(description='simple program for downloading a dataset to the local file location')
    parser.add_argument('--dataset', type=str, default='MNIST')
    return parser.parse_args(args=args)

if __name__ == '__main__':
    args = get_args()

    dataset = get_dataset(
        args.dataset, build=True,  
        dataset_parameters=dict(download=True)
    )

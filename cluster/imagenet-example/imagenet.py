import argparse
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import os
import sys
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from networkAlignmentAnalysis import datasets
from networkAlignmentAnalysis.models.registry import get_model


def configure_wandb(args):
    """create a wandb run file and set environment parameters appropriately"""
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            project=args.job_name,
            name=args.job_folder,
        )
        os.environ["WANDB_MODE"] = "offline"
        return run

    return None


def train(args, model, device, dataset, optimizer, epoch, rank, run, train=True):
    dataloader = dataset.train_loader if train else dataset.test_loader
    if dataset.distributed:
        if train:
            dataset.train_sampler.set_epoch(epoch)
        else:
            dataset.test_sampler.set_epoch(epoch)

    if rank == 0:
        first_batch_timer = time.time()

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        data, target = dataset.unwrap_batch(batch, device=device)
        if rank == 0 and batch_idx == 0:
            print(f"Train-- epoch {epoch}, rank {rank}, first batch loaded in {time.time() - first_batch_timer} seconds.")
        optimizer.zero_grad()
        output = model(data)
        loss = dataset.measure_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if rank == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(dataloader)} ({100.*batch_idx/len(dataloader):.0f}%)] \t Loss: {loss.item():.6f}")
                if run is not None:
                    run.log(dict(epoch=epoch, batch_idx=batch_idx, train_loss=loss.item()))
            if args.dry_run:
                break


def test(model, device, dataset, run, train=False):
    dataloader = dataset.train_loader if train else dataset.test_loader
    model.eval()
    test_loss = 0
    correct = 0
    attempts = 0
    with torch.no_grad():
        for batch in dataloader:
            data, target = dataset.unwrap_batch(batch, device=device)
            output = model(data)
            test_loss += dataset.measure_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            attempts += data.size(0)

    test_loss /= attempts

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{attempts} ({100.*correct/attempts:.0f}%)\n")

    if run is not None:
        run.log(dict(test_loss=test_loss, test_accuracy=100.0 * correct / attempts))


def create_dataset(name, net, distributed=True, loader_parameters={}):
    return datasets.get_dataset(
        name,
        build=True,
        distributed=distributed,
        transform_parameters=net,
        loader_parameters=loader_parameters,
    )


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Example")
    parser.add_argument(
        "--job-name",
        type=str,
        default="ImageNet-Example",
        help="job name for this project",
    )
    parser.add_argument(
        "--job-folder",
        type=str,
        default=".",
        help="job folder for storing data",
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="if used, will log experiment to WandB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--wd", type=float, default=0.0, metavar="WD", help="weight decay (default: 0.0)")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        metavar="M",
        help="Learning rate step gamma (default: 0.99)",
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()

    run = configure_wandb(args)

    torch.manual_seed(args.seed)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(
        f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" f" {gpus_per_node} allocated GPUs per node.",
        flush=True,
    )

    loader_parameters = dict(
        batch_size=args.batch_size,
        num_workers=max(int(os.environ["SLURM_CPUS_PER_TASK"]) - 1, 1),  # save a little headroom
    )

    if world_size > 1:
        setup(rank, world_size)
        if rank == 0:
            print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    model_name = "AlexNet"
    dataset_name = "ImageNet"
    net = get_model(model_name, build=True, dataset=dataset_name)
    dataset = create_dataset(dataset_name, net, distributed=world_size > 1, loader_parameters=loader_parameters)

    model = net.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank]) if world_size > 1 else model
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        epoch_time = time.time()
        train(args, ddp_model, local_rank, dataset, optimizer, epoch, rank, run)
        if rank == 0:
            test(ddp_model, local_rank, dataset, run)
        scheduler.step()
        epoch_time = time.time() - epoch_time
        if rank == 0:
            print(f"Epoch {epoch}, Train & Test Time = {epoch_time:.1f} seconds (measured from rank {rank}).\n")

    if args.save_model and rank == 0:
        torch.save(model.state_dict(), os.path.join(args.job_folder, "alexnet_imagenet.pt"))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

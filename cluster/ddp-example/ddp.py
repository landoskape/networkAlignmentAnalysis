"""
Massive thank you to the Princeton cluster engineers for providing this example (we've only made small changes)
https://github.com/PrincetonUniversity/multi_gpu_training/blob/main/02_pytorch_ddp/mnist_classify_ddp.py
"""

import argparse
import shutil
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, dataloader, datasampler, optimizer, epoch, rank):
    """basic training script"""
    # required for different shuffle order of examples in dataset each epoch
    datasampler.set_epoch(epoch)

    if rank == 0 and epoch == 1:
        first_batch_timer = time.time()

    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        if rank == 0 and epoch == 1 and batch_idx == 0:
            print(f"Train-- epoch {epoch}, rank {rank}, first batch loaded in {time.time() - first_batch_timer} seconds.")
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if rank == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(dataloader)} ({100.*batch_idx/len(dataloader):.0f}%)] \t Loss: {loss.item():.6f}")
            if args.dry_run:
                break


def test(model, device, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    attempts = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            attempts += data.size(0)

    test_loss /= attempts

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{attempts} ({100.*correct/attempts:.0f}%)\n")


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example with DDP")
    parser.add_argument(
        "--job-folder",
        type=str,
        default=".",
        help="job folder for storing data",
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
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        metavar="M",
        help="Learning rate step gamma (default: 0.9)",
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

    print("job folder:", args.job_folder)
    data_folder = os.path.join(args.job_folder, "data")

    torch.manual_seed(args.seed)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    print("gpus_per_node:", gpus_per_node)
    print("device_count:", torch.cuda.device_count())

    assert gpus_per_node == torch.cuda.device_count()
    print(
        f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" f" {gpus_per_node} allocated GPUs per node.",
        flush=True,
    )

    if world_size > 1:
        setup(rank, world_size)
        if rank == 0:
            print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    # Create network
    net = Net()
    model = net.to(local_rank)

    # Make it a DDP object for distributed processing and training
    ddp_model = DDP(model, device_ids=[local_rank]) if world_size > 1 else model
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST(data_folder, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_folder, train=False, download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
    )

    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            epoch_time = time.time()
        train(args, ddp_model, local_rank, train_loader, train_sampler, optimizer, epoch, rank)
        if rank == 0:
            test(ddp_model, local_rank, test_loader)
        scheduler.step()
        if rank == 0:
            epoch_time = time.time() - epoch_time
            print(f"\nEpoch {epoch}, Train & Test Time = {epoch_time:.1f} seconds (measured from rank {rank}).\n")

    if args.save_model and rank == 0:
        torch.save(model.state_dict(), os.path.join(args.job_folder, "test_model_ddp.pt"))

    if world_size > 1:
        dist.destroy_process_group()

    # clear locally downloaded MNIST data
    shutil.rmtree(data_folder)


if __name__ == "__main__":
    main()

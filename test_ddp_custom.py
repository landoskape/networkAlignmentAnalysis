import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import time
from multiprocessing import cpu_count
from networkAlignmentAnalysis import datasets
from networkAlignmentAnalysis.models.registry import get_model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def create_dataset(net):
    return datasets.get_dataset('MNIST', build=True, distributed=True, transform_parameters=net)

def train(dataset, model, criterion, optimizer, epoch, device, train=True):
    # switch to train mode
    model.train()
    dataloader = dataset.train_loader if train else dataset.test_loader
    datasampler = dataset.train_sampler if train else dataset.test_sampler

    if dataset.distributed:
        datasampler.set_epoch(epoch)

    end = time.time()
    for i, (images, target) in enumerate(dataloader):
        print('data loading time:', time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        print('total batch time:', time.time() - end)
        end = time.time()

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = get_model('MLP', build=True, dataset='MNIST').to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    dataset = create_dataset(model)

    num_epochs = 1
    for epoch in range(num_epochs):
        train(dataset, ddp_model, loss_fn, optimizer, epoch, device=f"cuda:{rank}", train=True)
    
    # cleanup
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f'num gpus:{n_gpus}')
    print(f'num cpus: {cpu_count()}')
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)



    # run_demo(demo_checkpoint, world_size)





# def demo_checkpoint(rank, world_size):
#     print(f"Running DDP checkpoint example on rank {rank}.")
#     setup(rank, world_size)

#     model = ToyModel().to(rank)
#     ddp_model = DDP(model, device_ids=[rank])


#     CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
#     if rank == 0:
#         # All processes should see same parameters as they all start from same
#         # random parameters and gradients are synchronized in backward passes.
#         # Therefore, saving it in one process is sufficient.
#         torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

#     # Use a barrier() to make sure that process 1 loads the model after process
#     # 0 saves it.
#     dist.barrier()

#     # configure map_location properly
#     map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#     ddp_model.load_state_dict(
#         torch.load(CHECKPOINT_PATH, map_location=map_location))

#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     optimizer.zero_grad()
#     outputs = ddp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(rank)

#     loss_fn(outputs, labels).backward()
#     optimizer.step()

#     # Not necessary to use a dist.barrier() to guard the file deletion below
#     # as the AllReduce ops in the backward pass of DDP already served as
#     # a synchronization.

#     if rank == 0:
#         os.remove(CHECKPOINT_PATH)

#     cleanup()

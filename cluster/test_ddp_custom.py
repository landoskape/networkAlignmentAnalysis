import os
import sys
import tempfile
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.profiler import profile, record_function, ProfilerActivity

import time
from multiprocessing import cpu_count
from networkAlignmentAnalysis import datasets
from networkAlignmentAnalysis.models.registry import get_model


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def create_dataset(name, net, distributed=True, loader_parameters={}):
    return datasets.get_dataset(
        name,
        build=True,
        distributed=distributed,
        transform_parameters=net,
        loader_parameters=loader_parameters,
    )


def train(dataset, model, optimizer, epoch, device, train=True):
    # switch to train mode
    model.train()
    dataloader = dataset.train_loader if train else dataset.test_loader
    datasampler = dataset.train_sampler if train else dataset.test_sampler

    if dataset.distributed:
        datasampler.set_epoch(epoch)

    start_time = time.time()
    for i, (images, target) in enumerate(dataloader):
        if i == 0:
            first_batch_time = time.time() - start_time
            print("training now, rank:", device, "first batch time:", first_batch_time)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = dataset.measure_loss(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == 10:
            break

    full_epoch_time = time.time() - start_time
    full_epoch_time -= first_batch_time
    full_epoch_time /= i
    print(
        "Epoch:",
        epoch,
        "Device:",
        device,
        "First batch:",
        first_batch_time,
        "Time per batch:",
        full_epoch_time,
    )


def demo(rank, world_size, distributed, num_epochs=2):
    if distributed:
        print(f"Running basic DDP example on rank {rank}.")
        setup(rank, world_size)
        get_device = lambda rank: f"cuda:{rank}"
    else:
        print("Running single GPU example.")
        get_device = lambda _: "cuda"

    model_name = "AlexNet"
    dataset_name = "CIFAR100"

    # create model and move it to GPU with id rank
    model = get_model(model_name, build=True, dataset=dataset_name).to(get_device(rank))
    print(rank, "model opened")
    ddp_model = DDP(model, device_ids=[rank]) if distributed else model
    print(rank, "ddp made")
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    if dataset_name == "ImageNet":
        loader_prms = {"batch_size": 64}
    else:
        loader_prms = {}
    loader_prms["num_workers"] = 4

    dataset = create_dataset(
        dataset_name, model, distributed=distributed, loader_parameters=loader_prms
    )
    print(rank, "dataset created")

    t = time.time()
    for epoch in range(num_epochs):
        print(rank, "starting epoch:", epoch)
        train(dataset, ddp_model, optimizer, epoch, device=get_device(rank), train=True)

    print(f"total time (ddp={distributed}):", time.time() - t)

    # cleanup
    if distributed:
        cleanup()


def run_demo(demo_fn, world_size, num_epochs):
    mp.spawn(demo_fn, args=(world_size, True, num_epochs), nprocs=world_size, join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"num gpus:{n_gpus}")
    print(f"num cpus: {cpu_count()}")
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    num_epochs = 15

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_setup:
    #    with record_function("setup profiler (don't report this one)"):
    #        demo(None, None, False, num_epochs=0)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_ddp:
    #    with record_function("DDP Example"):
    #        run_demo(demo, world_size, num_epochs)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_single:
    #    with record_function("Single GPU Example"):
    #        demo(None, None, False, num_epochs=num_epochs)

    # print('\n\n')
    # print(prof_ddp.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print('\n\n')
    # print(prof_single.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    run_demo(demo, world_size, num_epochs)
    demo(None, None, False, num_epochs=num_epochs)


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

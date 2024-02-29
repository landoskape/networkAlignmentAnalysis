import os
import sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

import time
from tqdm import tqdm

import torch
import torchvision
from torchvision.transforms import v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networkAlignmentAnalysis import files

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def check_time(num_workers, batch_size, pin_memory, fast_loader):
    transform = transforms.Compose(
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    )

    trainset = torchvision.datasets.CIFAR10(
        root=files.data_path(), train=True, download=False, transform=transform
    )

    loader_kwargs = dict(
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    if fast_loader:
        trainloader = torch.utils.data.DataLoader(
            trainset, **loader_kwargs, persistent_workers=fast_loader
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, **loader_kwargs, persistent_workers=fast_loader
        )

    net = Net()
    net.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    time_per_batch = torch.zeros((2, len(trainloader)))
    batch_time = time.time()
    for cycle in range(2):
        for idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            time_per_batch[cycle, idx] = time.time() - batch_time
            batch_time = time.time()

    avg_batch = torch.mean(time_per_batch[:, 1:])
    total_time = torch.sum(time_per_batch)
    init_time = time_per_batch[0, 0] - avg_batch  # estimate of prepare time
    second_init = time_per_batch[1, 0] - avg_batch
    return init_time, avg_batch, total_time, second_init


if __name__ == "__main__":
    print("using device: ", DEVICE)

    # tests
    num_workers = [0, 2, 4]
    batch_size = [8, 1024]
    pin_memory = [True, False]
    use_fast_loader = [True, False]

    for nw in num_workers:
        for bs in batch_size:
            for pin in pin_memory:
                for fast in use_fast_loader:
                    if nw == 0 and fast:
                        continue  # persistent memory is irrelevant when num_workers=0

                    init, avg, total, second = check_time(nw, bs, pin, fast)
                    print(f"NW={nw}, BS={bs}, Pin={pin}, Persistent={fast}")
                    print(
                        f"Dur={total:.3f}, PerBatch={avg:.2f}, Prep={init:.2f}, SecondPrep={second:.2f}"
                    )
                    print("")

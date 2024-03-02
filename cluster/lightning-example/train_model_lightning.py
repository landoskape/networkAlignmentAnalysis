from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import lightning as L

import argparse
import logging


class LightningModel(L.LightningModule):
    def __init__(self, model, optimizer, lr) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr

    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model(images)
        loss = torch.nn.functional.cross_entropy(output, labels)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer


# def train_one_epoch(model, loader, optimizer, device):
#     model.train()
#     batch_num = 0
#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         output = model(images)
#         loss = torch.nn.functional.cross_entropy(output, labels)
#         loss.backward()
#         optimizer.step()
#         if batch_num % 100 == 0:
#             logging.info(f"Batch Num: {batch_num} Loss: {loss.item()}")
#         batch_num += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--dataset", type=str, default="imagenet21k_train", help="dataset to use")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--model", type=str, default="alexnet", help="model to use")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus per node")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=0,
        help="number of training steps to save checkpoint after",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="directory to save checkpoints"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(args.dataset, transform=preprocess)
    logging.info(f"Data loaded, Found {len(dataset.classes)} classes, {len(dataset)} images")
    num_classes = len(dataset.classes)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    model = torchvision.models.get_model(args.model, num_classes=num_classes)
    optimizer = torch.optim.SGD
    lighting_model = LightningModel(model, optimizer, args.lr)

    callbacks = None
    if args.checkpoint_steps > 0:
        callbacks = [
            L.callbacks.ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                save_top_k=-1,
                save_last=True,
                every_n_train_steps=args.checkpoint_steps,
            )
        ]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        min_epochs=args.epochs,
        strategy="ddp",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        callbacks=callbacks,
    )
    trainer.fit(lighting_model, train_dataloaders=loader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

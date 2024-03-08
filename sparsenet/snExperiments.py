import os
import sys
from tqdm import tqdm
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sparsenet.atlSparseNet import SparseNet, argsSparseNet
from torch.utils.data import DataLoader
from sparsenet.ImageDataset import NatPatchDataset

from scipy.stats import normaltest
from sklearn.feature_selection import mutual_info_regression


def mainExperiment(opts={}):
    # save to tensorboard
    arg = argsSparseNet(opts)  # Defaults encoded in class
    # if use cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create net
    sparse_net = SparseNet(arg.n_neuron, arg.size, R_lr=arg.r_learning_rate, lmda=arg.reg, device=device)
    # load data
    batch_size = 500
    dataloader = DataLoader(
        NatPatchDataset(arg.batch_size, arg.size, arg.size, fpath="./data/IMAGES.mat"),
        batch_size=batch_size,
        shuffle=True,
    )
    # train
    optim = torch.optim.SGD([{"params": sparse_net.U.weight, "lr": arg.learning_rate}])

    # track loss & measure integration
    frac2remove = 0.25
    num2remove = int(arg.n_neuron * frac2remove)
    trackLoss = torch.zeros(arg.epoch * len(dataloader))
    woutHiAlignLoss = torch.zeros(arg.epoch * len(dataloader))
    woutLoAlignLoss = torch.zeros(arg.epoch * len(dataloader))
    alignMean = torch.zeros((arg.n_neuron, arg.epoch * len(dataloader)))
    batchIdx = 0
    for e in tqdm(range(arg.epoch)):
        running_loss = 0
        c = 0
        for img_batch in dataloader:  # tqdm(dataloader, desc='training', total=len(dataloader)):
            img_batch = torch.flatten(img_batch, 1).to(device)
            # update
            pred = sparse_net(img_batch)
            loss = ((img_batch - pred) ** 2).sum()
            running_loss += loss.item()
            loss.backward()
            # update U
            optim.step()
            # zero grad
            sparse_net.zero_grad()
            # norm
            sparse_net.normalize_weights()
            # save values
            trackLoss[batchIdx] = loss.item()
            alignMean[:, batchIdx] = sparse_net.measureAlignment(img_batch)
            idxCurrentAlign = torch.argsort(alignMean[:, batchIdx])
            idxWoutHi = idxCurrentAlign[:-num2remove]
            idxWoutLo = idxCurrentAlign[num2remove:]
            woutHiPred = sparse_net.forwardSubsetReuse(img_batch, idxWoutHi)
            woutLoPred = sparse_net.forwardSubsetReuse(img_batch, idxWoutLo)
            woutHiAlignLoss[batchIdx] = ((img_batch - woutHiPred) ** 2).sum().detach()
            woutLoAlignLoss[batchIdx] = ((img_batch - woutLoPred) ** 2).sum().detach()
            c += 1
            batchIdx += 1

    results = {
        "arg": arg,
        "sparse_net": sparse_net,
        "dataloader": dataloader,
        "optim": optim,
        "frac2remove": frac2remove,
        "trackLoss": trackLoss,
        "woutHiAlignLoss": woutHiAlignLoss,
        "woutLoAlignLoss": woutLoAlignLoss,
        "alignMean": alignMean,
    }
    return results

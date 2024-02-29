import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseNet(nn.Module):

    def __init__(self, K: int, M: int, R_lr: float = 0.1, lmda: float = 5e-3, device=None):
        super(SparseNet, self).__init__()
        self.K = K
        self.M = M
        self.R_lr = R_lr
        self.lmda = lmda
        # synaptic weights
        self.device = torch.device("cpu") if device is None else device
        self.U = nn.Linear(self.K, self.M**2, bias=False).to(self.device)
        # responses
        self.R = None
        self.normalize_weights()

    def ista_(self, img_batch):
        # create R
        self.R = torch.zeros((img_batch.shape[0], self.K), requires_grad=True, device=self.device)
        converged = False
        # update R
        optim = torch.optim.SGD([{"params": self.R, "lr": self.R_lr}])
        # train
        while not converged:
            old_R = self.R.clone().detach()
            # pred
            pred = self.U(self.R)
            # loss
            loss = ((img_batch - pred) ** 2).sum()
            loss.backward()
            # update R in place
            optim.step()
            # zero grad
            self.zero_grad()
            # prox
            self.R.data = SparseNet.soft_thresholding_(self.R, self.lmda)
            # convergence
            converged = torch.norm(self.R - old_R) / torch.norm(old_R) < 0.01

    def istaSubset_(self, img_batch, idx):
        # create R
        subk = len(idx)
        subR = torch.zeros((img_batch.shape[0], subk), requires_grad=True, device=self.device)
        subU = self.U.weight[:, idx].clone().detach()
        converged = False
        # update R
        optim = torch.optim.SGD([{"params": subR, "lr": self.R_lr}])
        # train
        while not converged:
            old_R = subR.clone().detach()
            # pred
            pred = (subU @ subR.T).T
            # loss
            loss = ((img_batch - pred) ** 2).sum()
            loss.backward()
            # update R in place
            optim.step()
            # zero grad
            subR.grad.zero_()
            # prox
            subR.data = SparseNet.soft_thresholding_(subR, self.lmda)
            # convergence
            converged = torch.norm(subR - old_R) / torch.norm(old_R) < 0.01
        return subR, subU

    @staticmethod
    def soft_thresholding_(x, alpha):
        with torch.no_grad():
            rtn = F.relu(x - alpha) - F.relu(-x - alpha)
        return rtn.data

    def zero_grad(self):
        self.R.grad.zero_()
        self.U.zero_grad()

    def normalize_weights(self):
        with torch.no_grad():
            self.U.weight.data = F.normalize(self.U.weight.data, dim=0)

    def forward(self, img_batch):
        # first fit
        self.ista_(img_batch)
        # now predict again
        pred = self.U(self.R)
        return pred

    def forwardSubset(self, img_batch, idx):
        subR, subU = self.istaSubset_(img_batch, idx)
        pred = (subU @ subR.T).T
        return pred

    def forwardSubsetReuse(self, img_batch, idx):
        useU = self.U.weight[:, idx].clone().detach()
        useR = self.R.data[:, idx].clone().detach()
        pred = (useU @ useR.T).T
        return pred

    def measureAlignment(self, img_batch, method="alignment"):
        weights = self.U.weight.data.detach()
        img_batch = torch.tensor(img_batch) if not torch.is_tensor(img_batch) else img_batch
        if method == "alignment":
            cc = torch.cov(img_batch.T)
        elif method == "similarity":
            cc = torch.corrcoef(img_batch.T)
            # need to add the 0 correction for pixels without variance!
            print("Add 0 correction for 0 variance pixels!!!")
        alignValue = torch.sum(torch.matmul(cc, weights) * weights, axis=0) / torch.trace(cc)
        return alignValue


class argsSparseNet(object):

    def __init__(self, userOpts={}):
        # Construct default opts dictionary
        opts = {}
        # model options
        opts["batch_size"] = 2000  # batch size
        opts["n_neuron"] = 400  # number of neurons
        opts["size"] = 10  # size of receptive fields
        # training options
        opts["epoch"] = 100  # number of epochs
        opts["learning_rate"] = 1e-2  # learning rate for weights
        opts["r_learning_rate"] = 1e-2  # learning rate for ISTA (activity on each image)
        opts["reg"] = 5e-3  # LSTM hidden size

        assert (
            userOpts.keys() <= opts.keys()
        ), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(opts.keys())}"
        opts.update(userOpts)

        for key in opts.keys():
            setattr(self, key, opts[key])


def initWeights(imEvec, imEval, numNeurons, scaleFunction=None):
    numEval = len(imEval)
    if scaleFunction is not None:
        imEval = scaleFunction(imEval)
    initBeta = np.random.exponential(scale=imEval, size=(numNeurons, numEval)).astype("float32")
    signBeta = np.random.randint(0, 2, size=initBeta.shape).astype("float32") * 2 - 1
    beta = torch.tensor(initBeta * signBeta)
    weights = imEvec @ beta.T
    return weights


def similarity(inputActivity, weights):
    # input activity is a (b x n) matrix, where b=batch and n=neurons
    # weights is a (m x n) matrix where each row corresponds to the weights for a single postsynaptic neuron
    # computes the rayleigh quotient R(cc, w) between the weight of each postsynaptic neuron and the correlation matrix of the inputs
    # then divides by n, because sum(eigenvalue)=sum(trace)=n, so this bounds the outputs between 0 and 1
    # -- note: this breaks if an input element has a standard deviation of 0! so we're just ignoring those values --
    inputActivity = (
        torch.tensor(inputActivity) if not torch.is_tensor(inputActivity) else inputActivity
    )
    weights = torch.tensor(weights) if not torch.is_tensor(weights) else weights
    idxMute = torch.where(torch.std(inputActivity, axis=0) == 0)[0]
    b, n = inputActivity.shape
    m = weights.shape[1]
    cc = torch.corrcoef(inputActivity.T)
    cc[idxMute, :] = 0
    cc[:, idxMute] = 0
    rq = torch.sum(torch.matmul(weights, cc) * weights, axis=1) / torch.sum(
        weights * weights, axis=1
    )
    return rq / n


def integration(sim):
    return -torch.log(sim)

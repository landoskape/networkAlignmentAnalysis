import numpy as np
import torch
from .experiment import Experiment
from ..models.registry import get_model, get_model_parameters
from . import arglib
from .. import utils
from .. import processing

from matplotlib import pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix


class LoadingPredictions(Experiment):
    def get_basename(self):
        """
        define basename for the LoadingPredictions experiment
        """
        return "LoadingPredictions"

    def prepare_path(self):
        """
        Define save location for each instance of this experiment type
        """
        return [self.args.comparison, self.args.network, self.args.dataset, self.args.optimizer]

    def make_args(self, parser):
        """
        Method for adding experiment specific arguments to the argument parser
        """
        parser = arglib.add_standard_training_parameters(parser)
        parser = arglib.add_network_metaparameters(parser)
        parser = arglib.add_checkpointing(parser)
        parser = arglib.add_dropout_experiment_details(parser)
        parser = arglib.add_alignment_analysis_parameters(parser)

        # add special experiment parameters
        # -- the "comparison" determines what should be compared by the script --
        # -- depending on selection, something about the networks are varied throughout the experiment --
        parser.add_argument("--comparison", type=str, default="lr")  # what comparison to do (see create_networks for options)
        parser.add_argument("--regularizers", type=str, nargs="*", default=["none", "dropout", "weight_decay"])
        parser.add_argument("--lrs", type=float, nargs="*", default=[1e-2, 1e-3, 1e-4])  # which learning rates to use

        # supporting parameters for some of the "comparisons"
        parser.add_argument("--compare-dropout", type=float, default=0.5)  # dropout when doing regularizer comparison
        parser.add_argument("--compare-wd", type=float, default=1e-5)  # weight-decay when doing regularizer comparison

        # return parser
        return parser

    # ----------------------------------------------
    # ------ methods for main experiment loop ------
    # ----------------------------------------------
    def create_networks(self):
        """
        method for creating networks

        depending on the experiment parameters (which comparison, which metaparams etc)
        this method will create multiple networks with requested parameters and return
        their optimizers and a params dictionary with the experiment parameters associated
        with each network
        """
        model_constructor = get_model(self.args.network)
        model_parameters = get_model_parameters(self.args.network, self.args.dataset)

        # get optimizer
        if self.args.optimizer == "Adam":
            optim = torch.optim.Adam
        elif self.args.optimizer == "SGD":
            optim = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer}) not recognized")

        # compare learning rates
        if self.args.comparison == "lr":
            lrs = [lr for lr in self.args.lrs for _ in range(self.args.replicates)]
            nets = [
                model_constructor(
                    dropout=self.args.default_dropout,
                    **model_parameters,
                    ignore_flag=self.args.ignore_flag,
                )
                for _ in lrs
            ]
            nets = [net.to(self.device) for net in nets]
            optimizers = [optim(net.parameters(), lr=lr, weight_decay=self.args.default_wd) for net, lr in zip(nets, lrs)]
            prms = {
                "lrs": lrs,  # the value of the independent variable for each network
                "name": "lr",  # the name of the parameter being varied
                "vals": self.args.lrs,  # the list of unique values for the relevant parameter
            }
            return nets, optimizers, prms

        # compare training with different regularizers
        elif self.args.comparison == "regularizer":
            dropout_values = [self.args.compare_dropout * (reg == "dropout") for reg in self.args.regularizers]
            weight_decay_values = [self.args.compare_wd * (reg == "weight_decay") for reg in self.args.regularizers]
            dropouts = [do for do in dropout_values for _ in range(self.args.replicates)]
            weight_decays = [wd for wd in weight_decay_values for _ in range(self.args.replicates)]
            nets = [model_constructor(dropout=do, **model_parameters, ignore_flag=self.args.ignore_flag) for do in dropouts]
            nets = [net.to(self.device) for net in nets]
            optimizers = [optim(net.parameters(), lr=self.args.default_lr, weight_decay=wd) for net, wd in zip(nets, weight_decays)]
            prms = {
                "dropouts": dropouts,  # dropout values by network
                "weight_decays": weight_decays,  # weight decay values by network
                "name": "regularizer",  # name of experiment
                "vals": self.args.regularizers,  # name of unique regularizers
            }
            return nets, optimizers, prms

        else:
            raise ValueError(f"Comparison={self.args.comparision} is not recognized")

    def main(self):
        """
        main experiment loop

        create networks (this is where the specific experiment is determined)
        train and test networks
        do supplementary analyses
        """
        # create networks
        nets, optimizers, prms = self.create_networks()

        # load dataset
        dataset = self.prepare_dataset(nets[0])

        # train networks
        train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)

        # do targeted dropout experiment
        dropout_results, dropout_parameters = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results["alignment"], train_set=False
        )

        # measure eigenfeatures
        _, _, eigenvector = utils.named_transpose([net.measure_eigenfeatures(dataset.test_loader) for net in nets])
        train_beta_by_class = [net.measure_class_eigenfeatures(dataset.train_loader, evecs) for net, evecs in zip(nets, eigenvector)]
        test_beta_by_class = [net.measure_class_eigenfeatures(dataset.test_loader, evecs) for net, evecs in zip(nets, eigenvector)]

        # make full results dictionary
        results = dict(
            prms=prms,
            train_results=train_results,
            test_results=test_results,
            train_beta_by_class=train_beta_by_class,
            test_beta_by_class=test_beta_by_class,
        )

        # return results and trained networks
        return results, nets

    def plot(self, results):
        """
        main plotting loop
        """
        self.plot_train_results(results["train_results"], results["test_results"], results["prms"])
        idx_layer = 2
        num_components = 10
        func = lambda x: torch.abs(x)
        num_to_average = 50

        # x_train/test composed of RMS loadings to a few images from the same class together
        x_train, x_test, y_train, y_test = self.get_train_test_rms_data(
            results["train_beta_by_class"],
            results["test_beta_by_class"],
            idx_layer,
            num_to_average,
        )

        # x_train/test composed of loadings for each image
        # x_train, x_test, y_train, y_test = get_train_test_data(train_beta_by_class, test_beta_by_class, idx_layer, func)

        ldas = self.fit_ldas(x_train, y_train, num_components)
        confusion_train, confusion_test, confusion_overall = self.test_ldas(ldas, x_train, y_train, x_test, y_test, num_components)
        print("Test Accuracy (within network):", np.diag(confusion_overall).reshape(3, 5))

        plt.close("all")
        fig, ax = plt.subplots(2, len(ldas), figsize=(9, 3), layout="constrained")
        for ii, (ctrain, ctest) in enumerate(zip(confusion_train, confusion_test)):
            ax[0, ii].imshow(ctrain, vmin=0, vmax=1, cmap="gray")
            ax[1, ii].imshow(ctest, vmin=0, vmax=1, cmap="gray")
            ax[0, ii].set_xticks([])
            ax[0, ii].set_yticks([])
            ax[1, ii].set_xticks([])
            ax[1, ii].set_yticks([])
        plt.show()

        plt.close("all")
        fig = plt.figure()
        plt.imshow(confusion_overall, vmin=0, vmax=1, cmap="gray")
        plt.colorbar()
        plt.show()

        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
        ax[0].imshow(confusion_train, vmin=0, vmax=1, cmap="gray")
        ax[1].imshow(confusion_test, vmin=0, vmax=1, cmap="gray")
        plt.show()

    def get_train_test_data(train_beta_by_class, test_beta_by_class, idx_layer, func=lambda x: x):
        num_classes = train_beta_by_class[0][0].size(0)
        num_per_class_train = train_beta_by_class[0][0].size(2)
        num_per_class_test = test_beta_by_class[0][0].size(2)
        x_train = [func(torch.concatenate([t for t in tbc[idx_layer]], dim=1).T) for tbc in train_beta_by_class]
        x_test = [func(torch.concatenate([t for t in tbc[idx_layer]], dim=1).T) for tbc in test_beta_by_class]
        y_train = [torch.repeat_interleave(torch.arange(num_classes), num_per_class_train) for _ in range(len(train_beta_by_class))]
        y_test = [torch.repeat_interleave(torch.arange(num_classes), num_per_class_test) for _ in range(len(train_beta_by_class))]
        return x_train, x_test, y_train, y_test

    def get_train_test_rms_data(train_beta_by_class, test_beta_by_class, idx_layer, num_average):
        """instead of putting all individual datapoints in, use RMS of num_average per sample"""
        num_classes, num_nodes, num_samples_train = train_beta_by_class[0][idx_layer].size()
        num_samples_test = test_beta_by_class[0][idx_layer].size(2)
        num_groups_train = int(num_samples_train / num_average)
        num_groups_test = int(num_samples_test / num_average)
        x_group_train = [
            tbc[idx_layer][:, :, : num_average * num_groups_train].reshape(num_classes, num_nodes, num_groups_train, num_average)
            for tbc in train_beta_by_class
        ]
        x_group_test = [
            tbc[idx_layer][:, :, : num_average * num_groups_test].reshape(num_classes, num_nodes, num_groups_test, num_average)
            for tbc in test_beta_by_class
        ]
        x_train = [torch.concatenate([x for x in utils.rms(xgt, 3)], dim=1).T for xgt in x_group_train]
        x_test = [torch.concatenate([x for x in utils.rms(xgt, 3)], dim=1).T for xgt in x_group_test]
        y_train = [torch.repeat_interleave(torch.arange(num_classes), num_groups_train) for _ in range(len(train_beta_by_class))]
        y_test = [torch.repeat_interleave(torch.arange(num_classes), num_groups_test) for _ in range(len(train_beta_by_class))]
        return x_train, x_test, y_train, y_test

    def fit_ldas(x_train, y_train, num_components):
        ldas = [LinearDiscriminantAnalysis() for _ in x_train]
        for ii, (xt, yt) in enumerate(zip(x_train, y_train)):
            ldas[ii].fit(xt[:, :num_components], yt)
        return ldas

    def test_ldas(ldas, x_train, y_train, x_test, y_test, num_components):
        num_nets = len(ldas)
        pred_train = [lda.predict(xt[:, :num_components]) for lda, xt in zip(ldas, x_train)]
        pred_test = [lda.predict(xt[:, :num_components]) for lda, xt in zip(ldas, x_test)]
        confusion_train = [confusion_matrix(yt.numpy(), pt) for yt, pt in zip(y_train, pred_train)]
        confusion_test = [confusion_matrix(yt.numpy(), pt) for yt, pt in zip(y_test, pred_test)]
        confusion_train = [ct / np.sum(ct, axis=1, keepdims=True) for ct in confusion_train]
        confusion_test = [ct / np.sum(ct, axis=1, keepdims=True) for ct in confusion_test]

        confusion_overall = np.zeros((num_nets, num_nets))
        for ii in range(num_nets):
            for jj in range(num_nets):
                c_pred = ldas[ii].predict(x_test[jj][:, :num_components])
                confusion_overall[ii, jj] = np.sum(c_pred == y_test[jj].numpy()) / len(y_test[jj])
        return confusion_train, confusion_test, confusion_overall

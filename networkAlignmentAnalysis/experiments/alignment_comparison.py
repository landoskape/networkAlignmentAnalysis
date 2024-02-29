import torch
from .experiment import Experiment
from ..models.registry import get_model, get_model_parameters
from . import arglib
from .. import processing
from .. import plotting


class AlignmentComparison(Experiment):
    def get_basename(self):
        """
        define basename for the AlignmentComparison experiment
        """
        return "alignment_comparison"

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
        parser.add_argument(
            "--comparison", type=str, default="lr"
        )  # what comparison to do (see load_networks for options)
        parser.add_argument(
            "--regularizers", type=str, nargs="*", default=["none", "dropout", "weight_decay"]
        )
        parser.add_argument(
            "--lrs", type=float, nargs="*", default=[1e-2, 1e-3, 1e-4]
        )  # which learning rates to use

        # supporting parameters for some of the "comparisons"
        parser.add_argument(
            "--compare-dropout", type=float, default=0.5
        )  # dropout when doing regularizer comparison
        parser.add_argument(
            "--compare-wd", type=float, default=1e-5
        )  # weight-decay when doing regularizer comparison

        # return parser
        return parser

    # ----------------------------------------------
    # ------ methods for main experiment loop ------
    # ----------------------------------------------
    def load_networks(self):
        """
        method for loading networks

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
            optimizers = [
                optim(net.parameters(), lr=lr, weight_decay=self.args.default_wd)
                for net, lr in zip(nets, lrs)
            ]
            prms = {
                "lrs": lrs,  # the value of the independent variable for each network
                "name": "lr",  # the name of the parameter being varied
                "vals": self.args.lrs,  # the list of unique values for the relevant parameter
            }
            return nets, optimizers, prms

        # compare training with different regularizers
        elif self.args.comparison == "regularizer":
            dropout_values = [
                self.args.compare_dropout * (reg == "dropout") for reg in self.args.regularizers
            ]
            weight_decay_values = [
                self.args.compare_wd * (reg == "weight_decay") for reg in self.args.regularizers
            ]
            dropouts = [do for do in dropout_values for _ in range(self.args.replicates)]
            weight_decays = [wd for wd in weight_decay_values for _ in range(self.args.replicates)]
            nets = [
                model_constructor(
                    dropout=do, **model_parameters, ignore_flag=self.args.ignore_flag
                )
                for do in dropouts
            ]
            nets = [net.to(self.device) for net in nets]
            optimizers = [
                optim(net.parameters(), lr=self.args.default_lr, weight_decay=wd)
                for net, wd in zip(nets, weight_decays)
            ]
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
        # load networks
        nets, optimizers, prms = self.load_networks()

        # load dataset
        dataset = self.prepare_dataset(nets[0])

        # train networks
        train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)

        # do targeted dropout experiment
        dropout_results, dropout_parameters = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results["alignment"], train_set=False
        )

        # measure eigenfeatures
        eigen_results = processing.measure_eigenfeatures(self, nets, dataset, train_set=False)

        # do targeted dropout experiment
        evec_dropout_results, evec_dropout_parameters = processing.eigenvector_dropout(
            self, nets, dataset, eigen_results, train_set=False
        )

        # make full results dictionary
        results = dict(
            prms=prms,
            train_results=train_results,
            test_results=test_results,
            dropout_results=dropout_results,
            dropout_parameters=dropout_parameters,
            eigen_results=eigen_results,
            evec_dropout_results=evec_dropout_results,
            evec_dropout_parameters=evec_dropout_parameters,
        )

        # return results and trained networks
        return results, nets

    def plot(self, results):
        """
        main plotting loop
        """
        plotting.plot_train_results(
            self, results["train_results"], results["test_results"], results["prms"]
        )
        plotting.plot_dropout_results(
            self,
            results["dropout_results"],
            results["dropout_parameters"],
            results["prms"],
            dropout_type="nodes",
        )
        plotting.plot_eigenfeatures(self, results["eigen_results"], results["prms"])
        plotting.plot_dropout_results(
            self,
            results["evec_dropout_results"],
            results["evec_dropout_parameters"],
            results["prms"],
            dropout_type="eigenvectors",
        )

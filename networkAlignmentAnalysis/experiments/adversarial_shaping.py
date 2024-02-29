import torch
from .experiment import Experiment
from ..models.registry import get_model, get_model_parameters
from . import arglib
from .. import processing
from .. import plotting
from ..utils import get_eval_transform_by_cutoff


class AdversarialShaping(Experiment):
    def get_basename(self):
        """
        define basename for the AdversarialShaping experiment
        """
        return "adversarial_shaping"

    def prepare_path(self):
        """
        Define save location for each instance of this experiment type
        """
        return [self.args.network, self.args.dataset, self.args.optimizer]

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
            "--cutoffs",
            type=float,
            nargs="*",
            default=[1e-2, 1e-3, 1e-4, 0.0],
            help="what fraction of total variance to cut eigenvalues off at",
        )
        parser.add_argument(
            "--manual-frequency",
            type=int,
            default=5,
            help="how frequently (by epoch) to do manual shaping with eigenvectors",
        )

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

        cutoffs = [co for co in self.args.cutoffs for _ in range(self.args.replicates)]
        nets = [
            model_constructor(
                dropout=self.args.default_dropout,
                **model_parameters,
                ignore_flag=self.args.ignore_flag,
            )
            for _ in cutoffs
        ]
        nets = [net.to(self.device) for net in nets]
        optimizers = [
            optim(net.parameters(), lr=self.args.default_lr, weight_decay=self.args.default_wd)
            for net in nets
        ]
        prms = {
            "cutoffs": cutoffs,  # the value of the independent variable for each network
            "name": "cutoff",  # the name of the parameter being varied
            "vals": self.args.cutoffs,  # the list of unique values for the relevant parameter
        }
        return nets, optimizers, prms

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
        special_parameters = dict(
            manual_shape=True,
            manual_frequency=self.args.manual_frequency,
            manual_transforms=[get_eval_transform_by_cutoff(co) for co in prms["cutoffs"]],
            manual_layers=nets[0].get_alignment_layer_indices(),
        )

        train_results, test_results = processing.train_networks(
            self, nets, optimizers, dataset, alignment=True, **special_parameters
        )

        # measure eigenfeatures
        eigen_results = processing.measure_eigenfeatures(self, nets, dataset, train_set=False)

        # do adversarial attack experiment
        adversarial_parameters = dict(
            epsilons=torch.linspace(0, 1, 11),
            use_sign=True,
            fgsm_transform=lambda x: x,
        )
        adversarial_results = processing.measure_adversarial_attacks(
            nets, dataset, self, eigen_results, train_set=False, **adversarial_parameters
        )

        # make full results dictionary
        results = dict(
            prms=prms,
            train_results=train_results,
            test_results=test_results,
            eigen_results=eigen_results,
            adversarial_results=adversarial_results,
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
        plotting.plot_eigenfeatures(self, results["eigen_results"], results["prms"])
        plotting.plot_adversarial_results(
            self, results["eigen_results"], results["adversarial_results"], results["prms"]
        )

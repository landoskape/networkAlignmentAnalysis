from datetime import datetime
from copy import copy
from argparse import ArgumentParser
from abc import ABC, abstractmethod 
from typing import Tuple, Dict, List
from numpy import load, save
from matplotlib.pyplot import savefig, show

from torch.nn import Module as TorchModule
from torch.cuda import is_available as cuda_available

from .. import files

class Experiment(ABC):
    def __init__(self) -> None:
        """Experiment constructor"""
        self.basename = self.get_basename() # Register basename of experiment
        self.basepath = files.results_path() / self.basename # Register basepath of experiment
        self.get_args() # Parse arguments to python program
        self.register_timestamp() # Register timestamp of experiment
        self.device = 'cuda' if cuda_available() else 'cpu'
        
    def report(self, init=False, args=False, meta_args=False) -> None:
        """Method for programmatically reporting details about experiment"""
        # Report general details about experiment 
        if init:
            print(f"Experiment object details:")
            print(f"basename: {self.basename}")
            print(f"basepath: {self.basepath}")
            print(f"experiment folder: {'/'.join(self.prepare_path())}")
            print('using device: ', self.device)

            # Report any other relevant details
            if self.args.save_networks and self.args.nosave:
                print("Note: setting nosave to True will overwrite save_networks. Nothing will be saved.")

        # Report experiment parameters
        if args:
            for key, val in vars(self.args).items():
                if key in self.meta_args: continue
                print(f"{key}={val}")

        # Report experiment meta parameters
        if meta_args:
            for key, val in vars(self.args).items():
                if key not in self.meta_args: continue
                print(f"{key}={val}")

    def register_timestamp(self) -> None:
        """
        Method for registering formatted timestamp.
        
        If timestamp not provided, then the current time is formatted and used to identify this particular experiment.
        If the timestamp is provided, then that time is used and should identify a previously run and saved experiment.
        """
        self.timestamp = self.args.timestamp if self.args.timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_dir(self, create=True):
        """
        Method for return directory of target file using prepare_path.
        """
        # exp_path is the base path followed by whatever folders define this particular experiment
        # (usually things like ['network_name', 'dataset_name', 'test', 'etc'])
        exp_path = copy(self.basepath)
        for app_dir in self.prepare_path():
            exp_path /= app_dir

        # use timestamp to save each run independently (or not to have a "master" run)
        if self.args.use_timestamp:
            exp_path = exp_path / self.timestamp

        # Make experiment directory if it doesn't yet exist
        if create and not(exp_path.exists()): 
            exp_path.mkdir(parents=True)

        return exp_path
    
    def get_path(self, name, create=True):
        """Method for returning path to file"""
        # get experiment directory
        exp_path = self.get_dir(create=create)

        # return full path (including stem)
        return exp_path / name
    
    @abstractmethod
    def get_basename(self) -> str:
        """Required method for defining the base name of the Experiment"""
        pass

    @abstractmethod
    def prepare_path(self) -> List[str]:
        """
        Required method for defining a pathname for each experiment.

        Must return a list of strings that will be appended to the base path to make an experiment directory.
        See ``get_dir()`` for details.
        """
        pass

    def get_args(self):
        """
        Method for defining and parsing arguments.
        
        This method defines the standard arguments used for any Experiment, and
        the required method make_args() is used to add any additional arguments
        specific to each experiment.
        """
        self.meta_args = [] # a list of arguments that shouldn't be updated when loading an old experiment
        parser = ArgumentParser(description=f"arguments for {self.basename}")
        parser = self.make_args(parser)
        
        # saving and new experiment loading parameters
        parser.add_argument('--nosave', default=False, action='store_true')
        parser.add_argument('--justplot', default=False, action='store_true')
        parser.add_argument('--save-networks', default=False, action='store_true')
        parser.add_argument('--showprms', default=False, action='store_true')

        # add meta arguments 
        self.meta_args += ['nosave', 'justplot', 'save_networks', 'showprms']
        
        # common parameters that shouldn't be updated when loading old experiment
        parser.add_argument('--use-timestamp', default=False, action='store_true')
        parser.add_argument('--timestamp', default=None, help='the timestamp of a previous experiment to plot or observe parameters')
        parser.add_argument('--showall', default=False, action='store_true', help='if true, will show all plots at once rather than having the user close each one for the next')

        # parse known arguments
        self.args = parser.parse_known_args()[0]
    
    @abstractmethod
    def make_args(self, parser) -> ArgumentParser:
        """
        Required method for defining special-case arguments.

        This should just use the add_argument method on the parser provided as input.
        """
        pass

    def get_prms_path(self):
        """Method for loading path to experiment parameters"""
        return self.get_dir() / 'prms.npy'
    
    def get_results_path(self):
        """Method for loading path to experiment results"""
        return self.get_dir() / 'results.npy'

    def _update_args(self, prms):
        """Method for updating arguments from saved parameter dictionary"""
        # First check if saved parameters contain unknown keys
        if prms.keys() > vars(self.args).keys():
            raise ValueError(f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(self.args).keys())}")
        
        # Then update self.args while ignoring any meta arguments
        for ak in vars(self.args):
            if ak in self.meta_args: continue # don't update meta arguments
            if ak in prms and prms[ak] != vars(self.args)[ak]:
                print(f"Requested argument {ak}={vars(self.args)[ak]} differs from saved, which is: {ak}={prms[ak]}. Using saved...")
                setattr(self.args, ak, prms[ak])

    def save_experiment(self, results):
        """Method for saving experiment parameters and results to file"""
        # Save experiment parameters
        save(self.get_prms_path(), vars(self.args))
        # Save experiment results
        save(self.get_results_path(), results)

    def load_experiment(self):
        """
        Method for loading saved experiment results.
        """
        # Check if it is there
        if not self.get_prms_path().exists():
            raise ValueError(f"saved parameters at: f{self.get_prms_path()} not found!")
        if not self.get_results_path().exists():
            raise ValueError(f"saved results at: f{self.get_results_path()} not found!")

        # Load parameters into object
        prms = load(self.get_prms_path(), allow_pickle=True).item()
        self._update_args(prms)
        
        # Load and return results
        return load(self.get_results_path(), allow_pickle=True).item()
    
    @abstractmethod
    def main(self) -> Tuple[Dict, List[TorchModule]]:
        """
        Required method for operating main experiment functions. 
        
        This method should perform any core training and analyses related to the experiment
        and return a results dictionary and a list of pytorch nn.Modules. The second requirement
        (torch modules) can probably be relaxed, but doesn't need to yet so let's keep it rigid.
        """
        pass

    @abstractmethod
    def plot(self, results: Dict) -> None:
        """
        Required method for operating main plotting functions.
        
        Should accept as input a results dictionary and run plotting functions.
        If any plots are to be saved, then each plotting function must do so 
        accordingly. 
        """
        pass

    def plot_ready(self, name):
        """method for saving and showing plot when it's ready"""
        # if saving, then save the plot
        if not self.args.nosave:
            savefig(str(self.get_path(name)))
        # show the plot now if not doing showall
        if not self.args.showall:
            show()



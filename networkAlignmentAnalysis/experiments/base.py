from datetime import datetime
from copy import copy
from argparse import ArgumentParser
from abc import ABC, abstractmethod 

from .. import files

class Experiment(ABC):
    def __init__(self):
        self.register_timestamp()
        self.basename = self.get_basename()
        self.basepath = files.results_path() / self.basename
        self.get_args()

        print(f"Experiment object created.")
        print(f"basename: {self.basename}")
        print(f"basepath: {self.basepath}")
        print(f"Saving results: {not self.args.nosave}")
        print(f"Saving networks: {self.args.save_networks}")

    def register_timestamp(self):
        """Method for returning formatted timestamp"""
        self.init_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_path(self, name):
        """Method for returning path to file using prepare_path directory and a file name"""
        exp_path = copy(self.basepath)
        for app_dir in self.prepare_path():
            exp_path /= app_dir

        # use timestamp to save each run independently (or not to have a "master" run)
        if self.args.use_timestamp:
            exp_path = exp_path / self.init_time

        # Make experiment directory if it doesn't yet exist
        if not exp_path.exists(): 
            exp_path.mkdir(parents=True)

        # return full path (including stem)
        return exp_path / name
    
    @abstractmethod
    def get_basename(self) -> str:
        """Required method for defining the base name of the Experiment"""
        pass

    @abstractmethod
    def prepare_path(self) -> list[str]:
        """
        Required method for defining a pathname for each experiment.

        Must return a list of strings that will be appended to the base path to make an experiment directory.
        """
        pass

    def get_args(self):
        """
        Method for defining and parsing arguments.
        
        This method defines the standard arguments used for any Experiment, and
        the required method make_args() is used to add any additional arguments
        specific to each experiment.
        """
        parser = ArgumentParser(description=f"arguments for {self.basename}")
        parser = self.make_args(parser)
        
        # saving parameters
        parser.add_argument('--nosave', default=False, action='store_true')
        parser.add_argument('--use-timestamp', default=False, action='store_true')
        parser.add_argument('--save-networks', default=False, action='store_true')

        # parse arguments
        self.args = parser.parse_args()
        
    @abstractmethod
    def make_args(self, parser) -> ArgumentParser:
        """
        Required method for defining special-case arguments.

        This should just use the add_argument method on the parser provided as input.
        """
        pass

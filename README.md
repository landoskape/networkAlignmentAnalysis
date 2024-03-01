# Network Alignment Analysis

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository is for a project to understand the structure of neural
networks with a method called "alignment". It contains modules which make
doing alignment-related experiments easy and the scripts that run the 
experiments. The repo is equipped to train pytorch models with DDP on an HPC
cluster. You'll find a few brief instructions about how to use the repository
here in the README, but for more information please feel free to reach out!

## Setup
The code requires a basic ML python environment. Setup can be done with a
standard python environment manager like conda (or mamba!). To get started,
clone the repository from GitHub, then navigate to the cloned folder. 

Create your environment with the following command. Mamba is much faster than
conda, but replace `mamba` with `conda` if you haven't installed `mamba`.
```
mamba env create -f environment.yml
mamba activate networkAlignmentAnalysis
```

If this doesn't work, it's probably because of the pytorch packages. Try 
commenting out the pytorch packages from the environment.yml file, then using
the above line to create and activate your environment. Type `nvidia-smi` in
your terminal to figure out the right CUDA version to use, and install them
using the instructions on the pytorch 
[website](https://pytorch.org/get-started/locally/).

There's no unit test, but to check if the install was successful, run the 
following script while in the environment and in the top directory:
```
python experiment.py alignment_stats --epochs 1 --replicates 1 --nosave
```

For checking HPC cluster related code, there's a MWE 
[here](cluster/ddp-example/), note that details will differ regarding the HPC
system etc.

### Local File Management
The repository depends on local file management to be established for many of
the core functions to be used. Specifically, a local folder containing 
standard machine learning datasets (like MNIST) and a file path designated for
saving any data created by the experiment scripts. 

To get this working, you'll need to edit ``networkAlignmentAnalysis/files.py`` 
file to include the directory containing your ML datasets and a results 
directory. To do so, pick a root directory and add a folder called "datasets"
and another one called "results" to it. Add the root directory to the 
dictionary called `PATH_REGISTRY`. The key should be whatever is returned by
``socket.gethostname()`` and the value should be the root folder on your 
computer. You'll also need to add whatever ML datasets you want to run 
experiments on. Right now it's only setup for MNIST, but support for more is
coming soon. Follow the directions on ML dataset websites and the convention
set by ``dataset_path()`` (found in ``files.py``) for full compatibility. 

## Usage and Tutorial
Detailed usage and tutorials can be found in a dedicated markdown file 
[here](media/usage.md).

## Contributing
Feel free to contribute to this project by opening issues or submitting pull
requests. It's already a collaborative project, so more minds are great if you
have ideas or anything to contribute!

## License
This project is licensed under the MIT License. If you use anything from this
repository for more than learning about code and/or pytorch, please cite us. 
There's no paper associated with the code at the moment, but you can cite our
GitHub repository URL or email us for any updates about this issue.





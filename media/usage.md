# Usage and Tutorials
Short of a documentation page, this is a simple md file that explains the main
features of the repository. For more information, please email or raise an 
issue. 

## Key Modules
There are a few key modules that wrap pytorch objects which make scripting 
experiments smooth and consistent. Essentially, the goal is to make a script
that performs an experiment look the same even if the experiment is performed
on neural networks with extremely different structures. There's a slight
learning curve and a little less flexibility but I think that's worth it for
the clear structure that it provides. 

### Alignment Networks
The package centers around a class called ``AlignmentNetwork``, which can be
found in ``networkAlignmentAnalysis/models/base.py``. This is a child of the
pytorch ``nn.Module`` class and is used to analyze artificial neural networks
with the alignment related tools developed for this project. 

The alignment analysis are performed between the input to a layer and a weight
matrix the input activity is projected onto (whether it's a feedforward layer,
a convolutional layer, or anything else). Any neural network layer with a 
weight layer you desire to study with alignment methods should be "registered"
as a layer in the ``AlignmentNetwork``. 

More to come ...


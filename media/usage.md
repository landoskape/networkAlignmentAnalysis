# Usage and Tutorials
Short of a documentation page, this is a simple md file that explains the main
features of the repository. For more information, please email or raise an 
issue. 

## Key Modules
There are a few key modules that wrap pytorch objects which make scripting 
experiments smooth and consistent. Essentially, the goal is to make a script
that performs an experiment look the same even if the experiment is performed
on neural networks with extremely different structures. 


The package centers around a class called ``AlignmentNetwork``, which can be
found in ``networkAlignmentAnalysis/models/base.py``. This is a child of the
pytorch ``nn.Module`` class and is used to analyze artificial neural networks
with the alignment related tools developed for this project. This class places
a few restrictions on network design, but that is desirable because it enables
the same code to be reused over and over again for the same analyses, even if
the analyses are performed on networks with extremely different structures. 
The DRY principle!

The alignment analysis are performed 


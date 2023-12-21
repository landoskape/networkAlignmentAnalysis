# Usage and Tutorials
Short of a documentation page, this is a simple md file that explains the main
features of the repository. For more information, please email or raise an 
issue. 

## Key Modules
The package centers around a class called ``AlignmentNetwork``, which can be
found in ``networkAlignmentAnalysis/models/base.py``. This is a child of the
pytorch ``nn.Module`` class and is used to analyze artificial neural networks
with the alignment related tools developed for this project. To make that work
it has a funny initialization structure 
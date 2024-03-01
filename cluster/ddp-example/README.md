## Distributed Data Parallel Example on Slurm Cluster

The contents of this folder contain a MWE for getting DDP to work on a cluster. If you want to try
it yourself, start from the main networkAlignmentAnalysis folder (with the .gitignore etc.). The
file called [ddp.slurm](ddp.slurm) is an ``sbatch`` script you can call with:

```shell
sbatch cluster/ddp-example/ddp.slurm
```

This prepares some environment variables, creates a job folder, then calls the python script 
[ddp.py](ddp.py) which downloads MNIST to the job folder, trains it across N GPUs (where N is 
equal to the number of nodes times the number of GPUs per node), and saves it if requested.

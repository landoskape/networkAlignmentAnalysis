#!/bin/bash
#SBATCH --job-name=ddp-example   # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=512G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

# note: 
# it's important for ntasks-per-node to be equal to the number of GPUs per node, which is set with --gres=gpu:4

export MASTER_PORT=12355 # there may be a smarter way to set this (e.g. with code), but this port is almost always open (maybe 100% of the time)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE)) # world size is equal to number of nodes and number of tasks per node
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load python
conda activate networkAlignmentAnalysis

start_time=$(date +%s)

# this is the command that initiates the processes
# srun will send each command to each task on each node for a total of WORLD_SIZE tasks
# (in this case the script it runs is python cluster/ddp.py)
srun python cluster/ddp.py --epochs=50 --dataset=ImageNet --model=AlexNet

end_time=$(date +%s)

total_time=$((end_time-start_time))
echo "Total Time= "$total_time" seconds"

#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1000G              # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

export MASTER_PORT=12355
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load python
conda activate networkAlignmentAnalysis

start_time=$(date +%s)

# (nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1 --filename=gpu_utilization.log)

srun python cluster/ddp.py --epochs=200 --dataset=ImageNet --model=AlexNet --log-interval=100 --save-model

end_time=$(date +%s)

total_time=$((end_time-start_time))
echo "Total Time= "$total_time" seconds"

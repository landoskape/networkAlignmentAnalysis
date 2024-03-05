#!/bin/bash
#SBATCH --job-name=ImageNet-Example     # create a short name for your job
#SBATCH --partition=kempner             # partition
#SBATCH --account=kempner_bsabatini_lab # account needed for kempner partition
#SBATCH --nodes=4                       # node count
#SBATCH --ntasks-per-node=4             # total number of tasks per node
#SBATCH --cpus-per-task=16              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4                    # number of allocated gpus per node
#SBATCH --mem=1000G                     # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=01:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send email when job begins
#SBATCH --mail-type=end                 # send email when job ends


# we need to define the job name directly since it isn't a slurm environment variable
JOB_NAME="ImageNet-Example" 

# this text file has some settings in it (like the standard job directory)
source cluster/slurm_settings.txt
JOB_DIR=${JOB_FOLDER}/${JOB_NAME}-${SLURM_JOB_ID} # make a specific directory for this particular job
mkdir -p $JOB_DIR

# define a unique log file in the right place
logfile="${JOB_DIR}/slurm-${SLURM_JOB_ID}.out"
echo "Writing to "$logfile

export MASTER_PORT=12355
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE >> $logfile
echo "MASTER_PORT="$MASTER_PORT >> $logfile

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR >> $logfile

module purge
module load python
conda activate networkAlignmentAnalysis

start_time=$(date +%s)

# (nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1 --filename=gpu_utilization.log)

srun python cluster/imagenet-example/imagenet.py --epochs=10 --use_wandb --log-interval=100 --job-folder=${JOB_DIR} --job-name=${JOB_NAME} --save-model >> $logfile

end_time=$(date +%s)

total_time=$((end_time-start_time))
echo "Total Time= "$total_time" seconds" >> $logfile
echo "Total Time= "$total_time" seconds"

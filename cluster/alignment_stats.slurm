#!/bin/bash
#SBATCH --job-name=alignment_stats      # create a short name for your job
#SBATCH --partition=kempner             # partition
#SBATCH --account=kempner_bsabatini_lab # account needed for kempner partition
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks-per-node=1             # total number of tasks per node
#SBATCH --cpus-per-task=64              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1                    # number of allocated gpus per node
#SBATCH --mem=1000G                     # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=04:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send email when job begins
#SBATCH --mail-type=end                 # send email when job ends

# we need to define the job name directly since it isn't a slurm environment variable
JOB_NAME="alignment_stats" 

# this text file has some settings in it (like the standard job directory)
source cluster/slurm_settings.txt
JOB_DIR=${JOB_FOLDER}/${JOB_NAME}-${SLURM_JOB_ID} # make a specific directory for this particular job
mkdir -p $JOB_DIR

# define a unique log file in the right place
logfile="${JOB_DIR}/slurm-${SLURM_JOB_ID}.out"
echo "Writing to ${logfile}"

# load python and activate our conda environment
module purge
module load python
conda activate networkAlignmentAnalysis

# record the start time
start_time=$(date +%s)

# this is the command that initiates the processes
python experiment.py alignment_stats --network CNN2P2 --dataset CIFAR100 --use_wandb --epochs 200 >> $logfile

# record the end time
end_time=$(date +%s)

# measure the time elapsed for the core part of the job in the logfile
total_time=$((end_time-start_time))
echo "Total Time= "$total_time" seconds" >> $logfile

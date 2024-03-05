#!/bin/bash
#SBATCH --job-name=test_files    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100M                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --output=/dev/null       # no output message

source cluster/slurm_settings.txt
mkdir -p $JOB_FOLDER
logfile="${JOB_FOLDER}/slurm-${SLURM_JOB_ID}.out"

echo "Writing to ${logfile}"
echo "Writing in ${logfile}" >> $logfile

module purge
module load python

start_time=$(date +%s)

python -c "print('hello world')" >> $logfile
python test_print.py >> $logfile

end_time=$(date +%s)

total_time=$((end_time-start_time))
echo "Total Time= "$total_time" seconds" >> $logfile

echo "Finished."

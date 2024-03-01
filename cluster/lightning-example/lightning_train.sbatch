#! /bin/bash
#SBATCH --job-name=vast_benchmark
#SBATCH --time=1-00:00:00
#SBATCH --partition=kempner_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1000G
#SBATCH --output=benchmark_lightning.out
#SBATCH --error=benchmark_lightning.err
#SBATCH --account=kempner_dev
#SBATCH --dependency=singleton
#SBATCH --gres=gpu:1

if [[ -z $1 ]] ; then
    model="alexnet"
else
     model=$1
fi


VAST_CREDENTIALS_PATH=/n/holylabs/LABS/kempner_dev/Users/$USER/vast-benchmarking/.load_credentials.sh
METRIC_GATHERING_SCRIPT_PATH=/n/holylabs/LABS/kempner_dev/Users/$USER/vast-benchmarking/get_vast_stats.sh
DATADIR=/n/fas-vast/kempner_benchmark/preproccessed_imagenet/imagenet21k_resized
CONTAINER_PATH=/n/holylabs/LABS/kempner_dev/Lab/containers/pytorch_2.1.2-cuda12.1-cudnn8-runtime-lightning.sif


source $VAST_CREDENTIALS_PATH
metrics_file=vast_metrics/vast_metrics_${SLURM_NNODES}nodes_${SLURM_GPUS_ON_NODE}gpus_${model}_${SLURM_JOB_ID}.csv

$METRIC_GATHERING_SCRIPT_PATH $metrics_file 30 &

time srun singularity exec --bind $DATADIR --nv $CONTAINER_PATH python3 train_model_lightning.py --dataset $DATADIR/imagenet21k_train \
     --epochs 1 --batch_size 64 --num_workers $SLURM_CPUS_PER_TASK --num_gpus $SLURM_GPUS_ON_NODE --num_nodes $SLURM_NNODES \
     --model $model
#! /bin/bash
# Script used to submit a matrix of jobs to the cluster

mkdir -p logs
mkdir -p vast_metrics

for model in alexnet resnet50 ; do
    for nodes in 1 2 4 8 16 32 ; do
        if [[ $nodes -eq 1 ]]; then
            for gpus in 1 2 4 ; do
                sbatch -N $nodes --ntasks-per-node $gpus --gres gpu:$gpus \
                -o logs/vast_benchmark_${nodes}nodes_${gpus}gpus_${model}.out \
                -e logs/vast_benchmark_${nodes}nodes_${gpus}gpus_${model}.err \
                lightning_train.sbatch $model 
                sleep 3
            done
        else
            gpus=4
            sbatch -N $nodes --ntasks-per-node $gpus --gres gpu:$gpus \
                -o logs/vast_benchmark_${nodes}nodes_${gpus}gpus_${model}.out \
                -e logs/vast_benchmark_${nodes}nodes_${gpus}gpus_${model}.err \
                lightning_train.sbatch $model 
                sleep 3
        fi
    done
done
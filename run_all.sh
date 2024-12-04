#!/bin/bash

# Arrays for learning rates and datasets
learning_rates=(1 0.8 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001)
datasets=("cifar10" "mnist" "fashionmnist")

# Iterate over all combinations of learning rates and datasets
for lr in "${learning_rates[@]}"
do
    for dataset in "${datasets[@]}"
    do
        echo "Running with learning rate: $lr, dataset: $dataset"
        python3 main.py --batch_size 64 --learning_rate $lr --optimizer all --epochs 100 --dataset $dataset
    done
done


import torch
from torch import nn, optim
from datetime import datetime
from itertools import product

# Import utils
from utils.Dataset import get_data_loader
from utils.Model import get_model
from utils.Optimizer import get_optimizer
from utils.Plot import plot_results
from utils.LoadSave import get_arguments, save_results
from Training import train_model, validate_model


def main():

    # Save current time for folder names
    start_time = datetime.now()

    # get configuration
    args, device = get_arguments()

    # Get all combinations of optimizers, learning rates, and datasets
    combinations = product(args.optimizer, args.learning_rate, args.dataset)

    # Iterate over all combinations
    for optimizer_name, learning_rate, dataset in combinations:
        print(f"\nRunning training with optimizer: {optimizer_name}, learning_rate: {learning_rate}, dataset: {dataset}")

        for run in range(args.runs):

            # load dataset
            train_loader, val_loader = get_data_loader(dataset, args.batch_size)

            # load model (do this for each optimizer to clear the model)
            model = get_model(dataset).to(device)

            # load model name for results
            model_name = model.__class__.__name__

            # load optimizer
            optimizer = get_optimizer(model, optimizer_name, learning_rate, device)

            # train model
            train_losses, val_losses, epoch_times = train_model(model, train_loader, val_loader, optimizer, device, args.epochs)

            # save results
            results_folder = save_results(start_time, run+1, args.runs, model, train_losses, val_losses, epoch_times, model_name, dataset, learning_rate, args.save_weights,  optimizer_name)

            # plot results
            plot_results(results_folder)


if __name__ == "__main__":
    main()


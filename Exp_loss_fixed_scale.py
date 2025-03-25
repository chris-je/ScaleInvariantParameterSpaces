import torch
from torch import nn, optim
from datetime import datetime
from itertools import product

# Import utils
from utils.Dataset import get_data_loader
from utils.Model import get_model
from utils.Optimizer import get_optimizer
from utils.Plot import plot_results, load_results, average_results, plot_lr_scale, median_results
from utils.LoadSave import get_arguments, save_results, check_results_exist
from utils.Training import train_model, validate_model
from parameterSpaces.FixModel import FixModel


"""
This is a modified version of main.py. It aims to create a 2D grid for learning rates and different factors for the fixed weights.
"""


def main():

    # Which weight to set the fixed weights
    scale_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]

    # Save current time for folder names
    start_time = datetime.now()

    # get configuration
    args, device = get_arguments()

    # Get all combinations of optimizers, learning rates, and datasets
    combinations = product(args.optimizer, args.learning_rate, args.dataset)

    # Iterate over all combinations
    for optimizer_name, learning_rate, dataset in combinations:

        # Iterate over fixed weights set
        for fixed_factor in scale_set:

            print(f"\nRunning training with optimizer: {optimizer_name}, learning_rate: {learning_rate}, dataset: {dataset}, fixed weights scale: {fixed_factor}")

            for run in range(args.runs):

                # load dataset
                train_loader, val_loader = get_data_loader(dataset, args.batch_size)

                # load model
                model = get_model(dataset, device)

                # load model name for results
                model_name = model.__class__.__name__

                # Fix model, if neccessary
                optimizer_name_new = optimizer_name
                optimizer_name_new = f"{optimizer_name} (Fixed weights) x {fixed_factor}"
                # Activate instead for no factor:
                fixed_number = 0
                # if fixed_factor:
                fixed_number = fixed_factor

                model = FixModel(model, device, fixed_factor).updated_model()

                # check if we even need to compute 
                results_exist = check_results_exist(run+1, args.runs, model_name, dataset, learning_rate, optimizer_name_new, args.output)
                if results_exist:
                    print("Skipping: Already computed")
                    continue

                # load optimizer
                optimizer = get_optimizer(model, optimizer_name, learning_rate, device)

                # train model
                train_losses, val_losses, epoch_times = train_model(model, train_loader, val_loader, optimizer, device, args.epochs)


                # save results
                results_folder = save_results( run+1, args.runs, model, train_losses, val_losses, epoch_times, model_name,
                                    dataset, learning_rate, args.save_weights,  optimizer_name_new, args.output,
                                    fixed_weight_size=fixed_number)


                # Plot 2D grid
                all_results = load_results(results_folder)
                results = median_results(all_results)
                plot_lr_scale(results, results_folder, "fixed_weight_size")


if __name__ == "__main__":
    main()


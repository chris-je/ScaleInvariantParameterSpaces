import torch
from torch import nn, optim
from datetime import datetime

# Import utils
from utils.Dataset import get_data_loader
from utils.Model import get_model
from utils.Optimizer import get_optimizer
from utils.Plot import PlotTrainingResults
from utils.LoadSave import get_arguments, save_results
from Training import train_model, validate_model



def main():

    results_folder = ""

    # get configuration
    args, device = get_arguments()

    # load dataset
    train_loader, val_loader = get_data_loader(args.dataset, args.batch_size)

    # load model
    model_name = get_model(args.dataset).__class__.__name__

    # load all optimizers
    optimizers = ["adam", "sgd", "gsgd"] if args.optimizer.lower() == "all" else [args.optimizer]

    # iterate over all optimizers if "all" was specified
    for optimizer_name in optimizers:

        # load model (do this for each optimizer to clear the model)
        model = get_model(args.dataset).to(device)

        # load optimizer
        optimizer = get_optimizer(model, optimizer_name, args.learning_rate, device)

        # train model
        train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device, args.epochs)

        # save results
        results_folder = save_results(results_folder, model, train_losses, val_losses, model_name, args.dataset, args.learning_rate, optimizer_name)

    # plot results
    plotter = PlotTrainingResults(results_folder)
    plotter.plot_results()

if __name__ == "__main__":
    main()


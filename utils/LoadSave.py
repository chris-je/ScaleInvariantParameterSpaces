import os
import torch
import argparse
import json
import argparse
from datetime import datetime
from pathlib import Path


"""
This file parses the input parameters when running the main.py.
Also, it allows us to check whether we have already computed a file and we can save the results in a json

Folder structure is generally as follows:
    results/
        {model} {dataset} {time}/
            {optimizer} {learning rate}/
                result1, result2, ...
"""


# parse arguments of the main script
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, nargs="+", default=[0.001])
    parser.add_argument("--optimizer", type=str, nargs="+", default=["sgd"])
    parser.add_argument("--dataset", type=str, nargs="+", default=["mnist"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--fix_weights", action="store_true")
    parser.add_argument("--use_gmodel", action="store_true")
    parser.add_argument("--output", type=str)

    # select graphic card if available
    device = (
        "cuda" if torch.cuda.is_available() else "mps"
        if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    # Save starting time for folder naming
    global start_time
    start_time = datetime.now()

    return parser.parse_args(), device


def save_results(run, total_runs, model, train_losses, val_losses, epoch_times,
                model_name, dataset_name, learning_rate, save_weights, optimizer_name, output=False,
                **extra_info):
    """
    Save training and validation losses in JSON file
    """
    
    # Create folder
    results_folder, base_folder = get_results_folder(model_name, dataset_name, learning_rate, optimizer_name, output, total_runs)
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # save model weights
    if save_weights:
        filename = f"weights_{optimizer_name}_lr={learning_rate}.pth" if total_runs == 1 else f"weights_run_{run}.pth"
        torch.save(model.state_dict(), os.path.join(results_folder, filename))

    # compute average step time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # save json with results
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "epochs": len(train_losses),
        "avg_epoch_time": avg_epoch_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
        **extra_info
    }
    
    filename = f"results_run_{run}.json"
    with open(os.path.join(results_folder, filename), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved in {results_folder}\n")

    return base_folder


def check_results_exist(run, total_runs, model_name, dataset_name, learning_rate, optimizer_name, output):
    """
    Return a boolean, whether the result we are trying to compute already exists
    """

    # Create / get folder
    results_folder, _ = get_results_folder(model_name, dataset_name, learning_rate, optimizer_name, output, total_runs)

    filename = (
        f"results_run_{run}.json"
    )
    
    file_path = os.path.join(results_folder, filename)
    return os.path.exists(file_path) and os.path.isdir(results_folder)


def get_results_folder(model_name, dataset_name, learning_rate, optimizer_name, output, total_runs):
    """
    Generates the folder path for our result file
    """
    base_folder = Path(f"results/{model_name}_{dataset_name}_{start_time.strftime('%Y_%m_%d_%H:%M:%S')}")
    if output:
        base_folder = Path(f"results/{output}")

    return base_folder / f"{optimizer_name}_lr={learning_rate}", base_folder
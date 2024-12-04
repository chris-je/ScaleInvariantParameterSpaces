import os
import torch
import argparse
import json
import argparse
from datetime import datetime


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

    # select corrent device
    device = (
        "cuda" if torch.cuda.is_available() else "mps"
        if torch.backends.mps.is_available() else "cpu"
    )
    # device = "cpu"
    print(f"Using {device} device")

    return parser.parse_args(), device

# save model weights and results in output folder
def save_results(start_time, run, total_runs, model, train_losses, val_losses, epoch_times, model_name, dataset_name, learning_rate, save_weights, optimizer_name):
    
    # Generate folder
    base_folder = f"results/{model_name}_{dataset_name}_{start_time.strftime('%Y_%m_%d_%H:%M:%S')}"
    results_folder = base_folder
    # Generate subfolder for runs
    if total_runs > 1:
        results_folder += f"/{optimizer_name}_lr={learning_rate}"
    os.makedirs(results_folder, exist_ok=True)
    

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
        "val_losses": val_losses
    }
    
    filename = f"results_{optimizer_name}_lr={learning_rate}.json" if total_runs == 1 else f"results_run_{run}.json"
    with open(os.path.join(results_folder, filename), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved in {results_folder}\n")

    return base_folder


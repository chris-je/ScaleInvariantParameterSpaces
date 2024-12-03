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
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="gsgd", help="adam, sgd, gsgd, or 'all' to run with each")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="mnist", help="fashionmnist, mnist, cifar10, cifar100")

    # select corrent device
    device = (
        "cuda" if torch.cuda.is_available() else "mps"
        if torch.backends.mps.is_available() else "cpu"
    )
    # device = "cpu"
    print(f"Using {device} device")

    return parser.parse_args(), device

# save model weights and results in output folder
def save_results(results_folder, model, train_losses, val_losses, model_name, dataset_name, learning_rate, optimizer_name):
    if results_folder == "":
        results_folder = f"results/{model_name}_{dataset_name}_{datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}"
        os.makedirs(results_folder, exist_ok=True)
    
    # save model weights
    torch.save(model.state_dict(), os.path.join(results_folder, f"weights_{optimizer_name}.pth"))

    # save json with results
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "epochs": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open(os.path.join(results_folder, f"results_{optimizer_name}.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved in {results_folder}\n")

    return results_folder


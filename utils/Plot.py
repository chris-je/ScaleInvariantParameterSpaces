import os
import json
import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_results(folder_path):
    all_results = load_results(folder_path)
    results = average_results(all_results)

    # Visualize speed of different optimizers
    plot_thoughput(results, folder_path)

    # Create plots for each optimizer showing multiple learning rates
    # Iterate through all the optimizers
    for optimizer_name, optimizer_results in results.items():
        # We got more than one learning rate for this optimizer => create plot
        if len(optimizer_results) > 1:
            plot_loss(optimizer_results, folder_path, "optimizer", optimizer_name)

    # Create plots for each learning rate showing multiple optimizers
    # Create set of learning rates and iterate through them
    learning_rates = set(lr for optimizer_results in results.values() for lr in optimizer_results.keys())
    for learning_rate in learning_rates:
        selected_results = {opt: res[learning_rate] for opt, res in results.items() if learning_rate in res}
        # We got more than more than one optimizer for this learning rate => create plot
        if len(selected_results) > 1:
            plot_loss(selected_results, folder_path, "learning_rate", learning_rate)
    

# load all jsons in the folder path
# TODO: refactor
def load_results(folder_path):
    # find all JSON files in the folder
    json_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    if not json_files:
        print(f"No JSON files found in folder: {folder_path}")
        return

    # load data from each JSON file
    results = {}
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Extract optimizer name and learning rate from folder structure
            subfolder = os.path.basename(os.path.dirname(file_path))
            if '_' in subfolder and '=' in subfolder:
                optimizer_name, lr_part = subfolder.split('_')
                learning_rate = lr_part.split('=')[1]
            else:
                # extract from filename if there are no subfolders (single run)
                filename = os.path.basename(file_path).replace('.json', '')
                if filename.startswith("results_") and '_lr=' in filename:
                    _, optimizer_name, lr_part = filename.split('_')
                    learning_rate = lr_part.split('=')[1]
                else:
                    optimizer_name = "unknown"
                    learning_rate = "unknown"
            
            if optimizer_name not in results:
                results[optimizer_name] = {}
            if learning_rate not in results[optimizer_name]:
                results[optimizer_name][learning_rate] = []
            results[optimizer_name][learning_rate].append(data)
    
    return results

# Average results from multiple runs
# TODO: refactor
def average_results(all_results):
    averaged_results = {}
    # Iterate through each optimizer
    for optimizer, lr_results in all_results.items():
        if optimizer not in averaged_results:
            averaged_results[optimizer] = {}
        # iterate through all files with the given optimizer
        for lr, runs in lr_results.items():
            if len(runs) > 1:
                avg_train_losses = np.mean([run['train_losses'] for run in runs], axis=0).tolist()
                avg_val_losses = np.mean([run['val_losses'] for run in runs], axis=0).tolist()
                avg_epoch_time = np.mean([run['avg_epoch_time'] for run in runs])
                averaged_results[optimizer][lr] = [{
                    "model": runs[0]['model'],
                    "dataset": runs[0]['dataset'],
                    "learning_rate": runs[0]['learning_rate'],
                    "optimizer": runs[0]['optimizer'],
                    "epochs": runs[0]['epochs'],
                    "avg_epoch_time": avg_epoch_time,
                    "train_losses": avg_train_losses,
                    "val_losses": avg_val_losses
                }]
            else:
                averaged_results[optimizer][lr] = runs
    return averaged_results

# Plot loss comparison between different optimizers
# def plot_loss(results, folder_path):
def plot_loss(results, folder_path, plot_type, identifier):

    # define a color cycle for repeatable colors
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    result_colors = {}

    # create a single plot for all optimizers
    plt.figure(figsize=(12, 8), dpi=200)  # Increase figure size and DPI for higher quality
    for key, result_list in results.items():
        # assign a color for the key (optimizer or learning rate) if not already assigned
        if key not in result_colors:
            result_colors[key] = next(color_cycle)

        for result in result_list:
            epochs = range(1, result['epochs'] + 1)
            train_losses = result['train_losses']
            val_losses = result['val_losses']
            color = result_colors[key]

            # plot training and validation losses
            label_prefix = f"lr={key} - " if plot_type == "optimizer" else f"{key} -"
            plt.plot(epochs, train_losses, linestyle='--', color=color, label=f"{label_prefix} Training Loss", linewidth=2)
            plt.plot(epochs, val_losses, linestyle='-', color=color, label=f"{label_prefix} Validation Loss", linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    title = f"Training and validation loss for {plot_type} {identifier}"
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # save plot in the folder
    plot_file_path = os.path.join(folder_path, f"loss_{plot_type}_{identifier}.png")
    plt.savefig(plot_file_path)
    plt.close()

# Plot throughput in iterations/s
def plot_thoughput(results, folder_path):
    # define a color cycle for repeatable colors
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    optimizer_colors = {}

    throughputs = {}
    for optimizer_name, learning_rate_results in results.items():
        avg_epoch_times = []
        for learning_rate, optimizer_results in learning_rate_results.items():
            avg_epoch_times.extend([result['avg_epoch_time'] for result in optimizer_results])
        if avg_epoch_times:
            avg_epoch_time = sum(avg_epoch_times) / len(avg_epoch_times)
            if avg_epoch_time > 0:
                throughputs[optimizer_name] = 1 / avg_epoch_time
            # assign a color for the optimizer if not already assigned
            if optimizer_name not in optimizer_colors:
                optimizer_colors[optimizer_name] = next(color_cycle)

    # create bar plot
    plt.figure(figsize=(12, 8), dpi=200)
    optimizer_names = list(throughputs.keys())
    throughput_values = list(throughputs.values())
    # assign colors
    bar_colors = [optimizer_colors[optimizer_name] for optimizer_name in optimizer_names]

    plt.bar(optimizer_names, throughput_values, color=bar_colors)
    plt.xlabel('Optimizers')
    plt.ylabel('Epochs per Second')
    plt.title('Throughput')
    plt.grid(axis='y')

    # save throughput plot in the folder
    plot_file_path = os.path.join(folder_path, "throughput.png")
    plt.savefig(plot_file_path)
    plt.close()




if __name__ == "__main__":
    # Example usage:
    folder_path = "results/MnistNet_mnist_2024_11_28_15:30:08"  # TODO: Replace with argument
    plot_results(folder_path)

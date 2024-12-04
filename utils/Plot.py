import os
import json
import matplotlib.pyplot as plt
import itertools

def plot_results(folder_path):

    results = load_results(folder_path)

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

    print(f"Plots have been saved in: {folder_path}")
    

# load all jsons in the folder path
def load_results(folder_path):
    # find all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in folder: {folder_path}")
        return

    # load data from each JSON file
    results = {}
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            optimizer_name = json_file.split('_')[1]  # Extract optimizer name from file name
            learning_rate = json_file.split('_')[2].split('-')[1].rstrip('.json') # Extract learning rate
            # TODO: 4
            if optimizer_name not in results:
                results[optimizer_name] = {}
            if learning_rate not in results[optimizer_name]:
                results[optimizer_name][learning_rate] = []
            results[optimizer_name][learning_rate].append(data)
    
    return results


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
    plt.title('Throughput (Epochs per Second)')
    plt.grid(axis='y')

    # save throughput plot in the folder
    plot_file_path = os.path.join(folder_path, "throughput.png")
    plt.savefig(plot_file_path)
    plt.close()




if __name__ == "__main__":
    # Example usage:
    folder_path = "results/MnistNet_mnist_2024_11_28_15:30:08"  # TODO: Replace with argument
    plot_results(folder_path)

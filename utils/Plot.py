import os
import json
import matplotlib.pyplot as plt
import itertools
import numpy as np
from tueplots import bundles, figsizes, fonts
import sys


"""
    Loads the data, averages it and computes a series of plots out of it.

    Requires the Latin Modern Math font to be installed.
"""



def plot_results(folder_path):
    all_results = load_results(folder_path)
    results = average_results(all_results)

    # Visualize computing speed of different optimizers
    plot_thoughput(results, folder_path)

    # Visualize loss for different lr
    plot_best_param_vs_loss(results, folder_path)

    # Selects the best learning rate for loss vs epochs
    plot_best_loss(results, folder_path, "learning_rate")
    

    # Create plots for each learning rate showing multiple optimizers
    learning_rates = set(lr for optimizer_results in results.values() for lr in optimizer_results.keys())
    for learning_rate in learning_rates:
        selected_results = {opt: res[learning_rate] for opt, res in results.items() if learning_rate in res}
        # We got more than more than one optimizer for this learning rate => create plot

        plot_loss2(selected_results, folder_path, "learning_rate", learning_rate)
    

# load all jsons in the folder path
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
                # Compute averages
                avg_train_losses = np.mean([run['train_losses'] for run in runs], axis=0).tolist()
                avg_val_losses = np.mean([run['val_losses'] for run in runs], axis=0).tolist()
                avg_epoch_time = np.mean([run['avg_epoch_time'] for run in runs])

                # Additional parameters
                additional_params = {key: runs[0][key] for key in runs[0] if key not in 
                                     ['train_losses', 'val_losses', 'avg_epoch_time']}

                # Create the averaged entry
                averaged_results[optimizer][lr] = [{
                    **additional_params,
                    "avg_epoch_time": avg_epoch_time,
                    "train_losses": avg_train_losses,
                    "val_losses": avg_val_losses
                }]
            else:
                averaged_results[optimizer][lr] = runs
    return averaged_results



def median_results(all_results):
    averaged_results = {}

    # Iterate through each optimizer
    for optimizer, lr_results in all_results.items():
        if optimizer not in averaged_results:
            averaged_results[optimizer] = {}

        # Iterate through all learning rates for the given optimizer
        for lr, runs in lr_results.items():
            if len(runs) > 1:
                # Convert NaNs to np.inf so they don't affect median calculations
                train_losses = np.array([run['train_losses'] for run in runs])
                val_losses = np.array([run['val_losses'] for run in runs])
                epoch_times = np.array([run['avg_epoch_time'] for run in runs])

                train_losses[np.isnan(train_losses)] = np.inf
                val_losses[np.isnan(val_losses)] = np.inf
                epoch_times[np.isnan(epoch_times)] = np.inf

                # Compute medians
                median_train_losses = np.nanmedian(train_losses, axis=0).tolist()
                median_val_losses = np.nanmedian(val_losses, axis=0).tolist()
                median_epoch_time = np.nanmedian(epoch_times)

                # Additional parameters (identical across runs)
                additional_params = {key: runs[0][key] for key in runs[0] if key not in 
                                     ['train_losses', 'val_losses', 'avg_epoch_time']}

                # Create the median entry
                averaged_results[optimizer][lr] = [{
                    **additional_params,
                    "avg_epoch_time": median_epoch_time,
                    "train_losses": median_train_losses,
                    "val_losses": median_val_losses
                }]
            else:
                averaged_results[optimizer][lr] = runs  # Single run case, keep as is

    return averaged_results

    
def plot_loss2(results, folder_path, plot_type, identifier):
    import itertools
    # Define optimizer information mapping
    optimizer_info = {
        "adam": {"display_name": "Adam", "color": "#1e75b4"},
        "sgd": {"display_name": "SGD", "color": "#ff7e0e"},
        "gadam": {"display_name": "G-Adam", "color": "#2ba02b"},
        "gsgd": {"display_name": "G-SGD", "color": "#d62728"},
        "fadam": {"display_name": "Fixed Adam", "color": "#2ba02b"},
        "fsgd": {"display_name": "Fixed SGD", "color": "#9267bd"},
        "adam (Fixed weights to 1)": {"display_name": "Fixed Adam", "color": "#2ba02b"},
        "sgd (Fixed weights to 1)": {"display_name": "Fixed SGD", "color": "#9267bd"},
        "sgd (Fixed weights)": {"display_name": "Fixed not 1 SGD", "color": "#9267bd"}
    }

    # Define a color cycle for fallback colors
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Tueplots
    plt.rcParams.update(bundles.icml2022(usetex=False, family="Latin Modern Math"))
    plt.rcParams.update({
        "figure.figsize": (0.6 * 14 / 2.54, 0.6 * 8 / 2.54),
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "legend.frameon": True,
    })

    # create a single plot for all optimizers
    for key, result_list in results.items():
        # Use optimizer_info if available, otherwise fall back to defaults
        if key in optimizer_info:
            display_name = optimizer_info[key]["display_name"]
            color = optimizer_info[key]["color"]
        else:
            display_name = key
            color = next(color_cycle)

        for result in result_list:
            epochs = range(0, result['epochs'])
            val_losses = result['val_losses']
            label_prefix = f"{display_name}" if plot_type == "optimizer" else f"{key}"

            # Plot only the validation loss
            plt.plot(epochs, val_losses, linestyle='-', color=color, label=label_prefix.strip(), linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    title = f"Training and validation loss for {plot_type} {identifier}"
    # plt.title(title)
    plt.legend()
    plt.grid(True)

    # save plot in the folder
    plot_file_path = os.path.join(folder_path, f"loss {plot_type} {identifier}")
    plt.savefig(f"{plot_file_path}.pdf", bbox_inches="tight")
    plt.close()


# best loss vs epoch
def plot_best_loss(results, folder_path, plot_type, identifier=None):
    # Define optimizer info mapping
    optimizer_info = {
        "adam": {"display_name": "Adam", "color": "#1e75b4"},
        "sgd": {"display_name": "SGD", "color": "#ff7e0e"},
        "gadam": {"display_name": "G-Adam", "color": "#2ba02b"},
        "gsgd": {"display_name": "G-SGD", "color": "#d62728"},
        "fadam": {"display_name": "Fixed Adam", "color": "#2ba02b"},
        "fsgd": {"display_name": "Fixed SGD", "color": "#9267bd"},
        "adam (Fixed weights to 1)": {"display_name": "Fixed Adam", "color": "#2ba02b"},
        "sgd (Fixed weights to 1)": {"display_name": "Fixed SGD", "color": "#9267bd"},
        "sgd (Fixed weights)": {"display_name": "Fixed not 1 SGD", "color": "#9267bd"}
    }

    # Define a color cycle for fallback colors
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    result_colors = {}

    # Tueplots style updates
    plt.rcParams.update(bundles.icml2022(usetex=False, family="Latin Modern Math"))
    plt.rcParams.update({
        "figure.figsize": (0.6 * 14 / 2.54, 0.6 * 8 / 2.54),
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "legend.frameon": True,
    })

    # Create a single plot for all optimizers
    for optimizer_name, optimizer_results in results.items():
        # Look up display name and color from optimizer_info, or fallback to defaults
        if optimizer_name in optimizer_info:
            display_name = optimizer_info[optimizer_name]["display_name"]
            color = optimizer_info[optimizer_name]["color"]
        else:
            display_name = optimizer_name
            if optimizer_name not in result_colors:
                result_colors[optimizer_name] = next(color_cycle)
            color = result_colors[optimizer_name]

        # When identifier is not provided and plot_type is "learning_rate",
        # assume optimizer_results is a dict mapping learning rates to lists of runs.
        if plot_type == "learning_rate" and identifier is None:
            best_lr = None
            best_loss = float('inf')
            best_result_list = None

            # Iterate over learning rates for this optimizer
            for lr, result_list in optimizer_results.items():
                # Compute the average of best validation losses over all runs for this lr
                best_losses = [np.min(result['val_losses']) for result in result_list]
                avg_best_loss = np.mean(best_losses)
                if avg_best_loss < best_loss:
                    best_loss = avg_best_loss
                    best_lr = lr
                    best_result_list = result_list

            # Use the selected best learning rate's results
            result_list_to_plot = best_result_list
            label_prefix = f"{display_name} (lr={best_lr})"
        else:
            # If an identifier is provided, assume optimizer_results is already a list of runs.
            result_list_to_plot = optimizer_results
            label_prefix = f"{display_name}"

        # Plot only the validation loss for the selected results
        for result in result_list_to_plot:
            epochs = range(result['epochs'])
            val_losses = result['val_losses']
            plt.plot(epochs, val_losses, linestyle='-', color=color,
                     label=label_prefix, linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    if identifier is not None:
        title = f"Validation loss for {plot_type} {identifier}"
        plot_file_path = os.path.join(folder_path, f"best_loss_{plot_type}_{identifier}")
    else:
        title = "Validation loss (best learning rate selection)"
        plot_file_path = os.path.join(folder_path, f"best_loss_{plot_type}")
    # plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{plot_file_path}.pdf", bbox_inches="tight")
    plt.close()



def plot_thoughput(results, folder_path):
    """
    Compare the speed of different optimizers by plotting the throughput in iterations/sec
    """

    # define a color cycle for repeatable colors
    custom_colors = ['#d95f02', '#1b9e77', '#7570b3', '#d62728', '#9467bd', '#8c564b']
    color_cycle = itertools.cycle(custom_colors)
    optimizer_colors = {}

    # Change text and font settings
    plt.rcParams['text.color'] = '#333'  # Set all text color to gray
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10  # Set global font size for axes ticks

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

    # Set axis labels and title
    plt.xlabel('Optimizers', fontsize=14, color='#333')  # Axis description font size 14
    plt.ylabel('Epochs per Second', fontsize=14, color='#333')  # Axis description font size 14
    plt.title('Throughput', fontname="DejaVu Sans", size=14, pad=26, color='#333')

    # Customize axes ticks
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=10, colors='#333')  # Set tick font size and color
    ax.tick_params(axis='y', labelsize=10, colors='#333')  # Set tick font size and color
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')

    # Remove vertical grid lines and add extra horizontal grid lines
    plt.grid(axis='x', which='both', linestyle='', linewidth=0)  # Remove vertical grid lines
    plt.grid(axis='y', which='major', linestyle='-', linewidth=1, alpha=0.8)  # Major horizontal grid lines
    plt.grid(axis='y', which='minor', linestyle='-', linewidth=0.6, alpha=0.5)  # Minor horizontal grid lines
    plt.minorticks_on()  # Enable minor ticks on the y-axis

    # Save throughput plot in the folder
    plot_file_path = os.path.join(folder_path, "throughput.png")
    plt.savefig(plot_file_path)
    plt.close()


def plot_best_param_vs_loss(results, folder_path, x_param="learning_rate"):
    """
    Plots the loss in comparison to another parameter. Chooses the epoch with the best overall validation loss
    """

    # Define a color cycle for fallback colors
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Mapping of optimizer names to display names and colors
    optimizer_info = {
        "adam": {"display_name": "Adam", "color": "#1e75b4"},
        "sgd": {"display_name": "SGD", "color": "#ff7e0e"},
        "gadam": {"display_name": "G-Adam", "color": "#2ba02b"},
        "gsgd": {"display_name": "G-SGD", "color": "#d62728"},
        "fadam": {"display_name": "Fixed Adam", "color": "#2ba02b"},
        "fsgd": {"display_name": "Fixed SGD", "color": "#9267bd"},
        "adam (Fixed weights to 1)": {"display_name": "Fixed Adam", "color": "#2ba02b"},
        "sgd (Fixed weights to 1)": {"display_name": "Fixed SGD", "color": "#9267bd"},
        "sgd (Fixed weights)": {"display_name": "Fixed not 1 SGD", "color": "#9267bd"}
    }

    # Update plot style settings
    plt.rcParams.update(bundles.icml2022(usetex=False, family="Latin Modern Math"))
    plt.rcParams.update({
        "figure.figsize": (0.6 * 14 / 2.54, 0.6 * 8 / 2.54),
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "legend.frameon": True,
    })

    for optimizer_name, param_results in results.items():
        # Use optimizer_info if available, otherwise fall back to defaults
        if optimizer_name in optimizer_info:
            display_name = optimizer_info[optimizer_name]["display_name"]
            color = optimizer_info[optimizer_name]["color"]
        else:
            display_name = optimizer_name
            color = next(color_cycle)

        x_values = []
        avg_val_losses = []
        
        for param_value, optimizer_results in param_results.items():
            # Skip entries that do not contain the requested parameter
            if x_param not in optimizer_results[0]:
                continue
            
            # Compute the best (minimum) validation loss per run and average them
            best_val_losses = [np.min(result['val_losses']) for result in optimizer_results]
            avg_best_val_loss = np.mean(best_val_losses)

            x_values.append(float(param_value))
            avg_val_losses.append(avg_best_val_loss)

        # Sort the values for a cleaner plot
        if x_values:
            sorted_indices = np.argsort(x_values)
            x_values = np.array(x_values)[sorted_indices]
            avg_val_losses = np.array(avg_val_losses)[sorted_indices]

            plt.plot(x_values, avg_val_losses, marker='o', linestyle='-', linewidth=2,
                     label=display_name, color=color)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(x_param.replace("_", " ").title())
    plt.ylabel('Loss')
    plt.legend()
    
    # Show grid only for major ticks (sparse intervals)
    plt.grid(True, which="major", linestyle="-", linewidth=0.5)

    # Save the plot
    plot_file_path = os.path.join(folder_path, f"best loss {x_param}")
    plt.savefig(f"{plot_file_path}.pdf", bbox_inches="tight")
    plt.close()



def plot_lr_scale(results, folder_path, custom_param):
    """
    Creates a 2D grid plot using learning rates and a custom parameter for all optimizers.

    The x-axis shows learning rates (log scale), the y-axis shows the provided custom parameter,
    and each cell is colored based on the average validation loss.

    Parameters
    ----------
    results : dict
        Dictionary mapping optimizer names to results. Each optimizer name maps
        to a dict mapping learning rate strings to a list of result dicts.
    folder_path : str
        Path where the plots will be saved.
    custom_param : str
        The key in the JSON where the custom parameter is stored (e.g., "fixed_weight_size").
    """

    plt.rcParams.update(bundles.icml2022(usetex=False, family="Latin Modern Math"))
    plt.rcParams.update({
        "figure.figsize": (7 / 2.54, 4 / 2.54),
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "legend.frameon": True,
    })

    all_entries = []  # Collects all (learning_rate, custom_param_value, avg_val_loss)

    for optimizer_name, lr_results in results.items():
        for learning_rate, optimizer_results in lr_results.items():
            for result in optimizer_results:
                # Ensure the custom parameter exists in the result
                if custom_param not in result:
                    continue  # Skip if missing
                
                custom_value = result[custom_param]  # Extract custom parameter value
                avg_val_loss = np.mean(result['val_losses']) if 'val_losses' in result else None
                
                if avg_val_loss is not None:
                    all_entries.append((float(learning_rate), float(custom_value), avg_val_loss))

    # Ensure we have data to plot
    if not all_entries:
        print("No valid data found for plotting.")
        return

    # Convert to NumPy array (N, 3)
    entries = np.array(all_entries)
    lrs = entries[:, 0]
    custom_values = entries[:, 1]
    losses = entries[:, 2]

    unique_lr = np.unique(lrs)
    unique_custom = np.unique(custom_values)

    # Build a grid: rows = custom parameter values, columns = learning rates
    grid = np.full((len(unique_lr), len(unique_custom)), np.nan)
    for lr, custom, loss in entries:
        lr_idx = np.searchsorted(unique_lr, lr)
        custom_idx = np.searchsorted(unique_custom, custom)

        grid[lr_idx, custom_idx] = loss

    # Compute cell edges for pcolormesh
    lr_step = (unique_lr[1] - unique_lr[0]) if len(unique_lr) > 1 else (unique_lr[0] / 2 if unique_lr[0] > 0 else 0.1)
    custom_step = (unique_custom[1] - unique_custom[0]) if len(unique_custom) > 1 else (unique_custom[0] / 2 if unique_custom[0] > 0 else 0.1)

    lr_edges = np.concatenate(([unique_lr[0] - lr_step / 2], unique_lr + lr_step / 2))
    custom_edges = np.concatenate(([unique_custom[0] - custom_step / 2], unique_custom + custom_step / 2))

    # Swap x/y
    lr_edges, custom_edges = custom_edges, lr_edges


    # Create grid plot using imshow for square boxes
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap='RdYlGn_r', origin='lower', aspect='equal')
    fig.colorbar(im, ax=ax, label="Loss")


    # Set tick positions and labels
    ax.set_yticks(np.arange(len(unique_lr)))
    ax.set_xticks(np.arange(len(unique_custom)))
    ax.set_yticklabels([f"{lr:.2e}" for lr in unique_lr])
    ax.set_yticks(ax.get_yticks()[::2])  # Select every 2nd tick, skipping the first
    ax.set_xticks(ax.get_xticks()[1::2])  # Select every 2nd tick, skipping the first
    ax.set_xticklabels([f"{val:.1f}" for val in unique_custom[1::2]])  # Round to 1 decimal

    # Set full tick positions
    yticks = np.arange(len(unique_lr))
    xticks = np.arange(len(unique_custom))

    # Keep every second tick
    yticks = yticks[::4]
    xticks = xticks[1::4]

    # Apply them
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)

    # Format ticks
    ax.set_yticklabels([f"$10^{{{int(np.log10(unique_lr[i]))}}}$" for i in yticks])
    ax.set_xticklabels([f"{unique_custom[i]:.1f}" for i in xticks])

    ax.set_ylabel("Log lr")
    ax.set_xlabel(custom_param.replace('_', ' '))  # Format label dynamically

    # Save as a single file
    plot_file_path = os.path.join(folder_path, "2D grid")
    plt.savefig(f"{plot_file_path}.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    # Ensure a folder name is provided
    if len(sys.argv) != 2:
        print("Usage: python3 Plot.py <folder_name>, ")
        sys.exit(1)

    folder_name = sys.argv[1]
    folder_path = f"results/{folder_name}/"


    plot_results(folder_path)

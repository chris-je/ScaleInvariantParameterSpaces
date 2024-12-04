import os
import json
import matplotlib.pyplot as plt
import itertools

class PlotTrainingResults:

    def plot_results(self, folder_path):
        self.folder_path = folder_path

        results = self.load_results()
        self.plot_loss(results)
        self.plot_thoughput(results)

        print(f"Plots have been saved in: {self.folder_path}")
        

    # load all jsons in the folder path
    def load_results(self):
        # find all JSON files in the folder
        json_files = [f for f in os.listdir(self.folder_path) if f.endswith('.json')]
        if not json_files:
            print(f"No JSON files found in folder: {self.folder_path}")
            return

        # load data from each JSON file
        results = {}
        for json_file in json_files:
            file_path = os.path.join(self.folder_path, json_file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                optimizer_name = json_file.split('_')[1]  # Extract optimizer name from file name
                if optimizer_name not in results:
                    results[optimizer_name] = []
                results[optimizer_name].append(data)
        
        return results


    # Plot loss comparison between different optimizers
    def plot_loss(self, results):

        # define a color cycle for repeatable colors
        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        optimizer_colors = {}

        # create a single plot for all optimizers
        plt.figure(figsize=(12, 8), dpi=200)  # Increase figure size and DPI for higher quality
        for optimizer_name, optimizer_results in results.items():
            # assign a color for the optimizer if not already assigned
            if optimizer_name not in optimizer_colors:
                optimizer_colors[optimizer_name] = next(color_cycle)

            for result in optimizer_results:
                epochs = range(1, result['epochs'] + 1)
                train_losses = result['train_losses']
                val_losses = result['val_losses']
                color = optimizer_colors[optimizer_name]

                # plot training and validation losses
                plt.plot(epochs, train_losses, linestyle='--', color=color, label=f"{optimizer_name} - Training Loss", linewidth=2)
                plt.plot(epochs, val_losses, linestyle='-', color=color, label=f"{optimizer_name} - Validation Loss", linewidth=2)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title(f"Training and validation loss for all optimizers")
        plt.legend()
        plt.grid(True)

        # save plot in the folder
        plot_file_path = os.path.join(self.folder_path, "combined_loss.png")
        plt.savefig(plot_file_path)
        plt.close()

    # Plot throughput in iterations/s
    def plot_thoughput(self, results):

        # define a color cycle for repeatable colors
        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        optimizer_colors = {}

        throughputs = {}
        for optimizer_name, optimizer_results in results.items():
            # compute throughput with 1/avg_epoch_time for each optimizer
            avg_epoch_times = [result['avg_epoch_time'] for result in optimizer_results]
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
        plt.ylabel('Steps per Second')
        plt.title('Throughput (Steps per Second) for Each Optimizer')
        plt.grid(axis='y')

        # save throughput plot in the folder
        plot_file_path = os.path.join(self.folder_path, "throughput.png")
        plt.savefig(plot_file_path)
        plt.close()



if __name__ == "__main__":
    # Example usage:
    folder_path = "results/MnistNet_mnist_2024_11_28_15:30:08"  # TODO: Replace with argument
    plotter = PlotTrainingResults(folder_path)
    plotter.plot_results()

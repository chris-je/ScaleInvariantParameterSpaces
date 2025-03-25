import torch
import torch.nn as nn
import time
import math
from scipy.optimize import linear_sum_assignment


def reparametrize(model):
    """
    Reparametrize a ReLU model in order to only have 1 and -1 as values on the diagonals of the weight matrices
    (except for the first layer)
    while keeping the functionality the same.

    Constraints:
        - All Layers have to be fully connected linear layers
        - All Activation functions have to be ReLU
    """
    D, D_inv = create_diagonal_matrices(model)
    return apply_reparam(model, D, D_inv)



def create_diagonal_matrices(model):
    """
    Create scaling matrices for a ReLU model. They are optimized using the Hungarian algorithm
    Starting with the last layer.
    Essentially computing D[i-1] = | diag(D[i] W[i]) |
    and it's inverse D⁻¹[i-1]
    """

    # Initialize layers and scaling matrices
    layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    N = len(layers)
    if N == 0:
        return []
    D_inv = [None] * (N - 1)
    D = [None] * (N - 1)

    
    # Iterate through layers in reverse
    for i in range(N - 1, 0, -1):
        W = layers[i].weight.data

        # Check if last layer => D doesn't exist
        if i == N-1:
            DW = W
        else:
            DW = D[i] @ W

        # D @ W not square => make square out of it
        if DW.shape[0] < DW.shape[1]:
            # Repeat matrix until it is square
            biggerSize = max(DW.shape[0], DW.shape[1])
            repeats = (math.ceil(biggerSize / DW.shape[0]), math.ceil(biggerSize / DW.shape[1]))
            DW = torch.tile(DW, repeats)[:biggerSize, :biggerSize]

        # Use this, if we want to use the diagonal as skeleton weights
        # Diagonal entries of D @ W
        diag = DW.diag()

        # Select best entries for skeleton weights (the ones closest to 1 or -1)
        # Compute cost for each weight, if it is used as skeleton
        cost_matrix = torch.min(torch.abs(DW - 1), torch.abs(DW + 1)).numpy()
        # Find indices of best skeleton candidates
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        selected_values = DW[row_ind, col_ind]
        diag[col_ind] = selected_values

        # Use this, if we want to use the diagonal as skeleton weights
        # Diagonal entries of D @ W
        # diag = DW.diag()

        # Check if matrix is valid
        if torch.any(diag == 0):
            raise ValueError(f"Layer '{i}' has 0-entries on the diagonal.")

        # Compute next D and D^-1
        D_inv[i-1] = torch.abs(torch.diag(1.0 / diag))
        D[i-1] = torch.abs(torch.diag(diag))
    
    return D, D_inv


def apply_reparam(model, D, D_inv):
    """
    Reparametrize a model given the matrices D and D⁻¹

    Args:
        D:      List of matrixes to multiply from the left (omitting the last layer): DW
        D^-1:   List of matrices to multiply from the right (starting at the second layer): WD^-1
    Returns:
        model:  Reparametrized model with same functionality
    """

    # Create new instance of the model
    new_model = model.__class__()
    new_model.load_state_dict(model.state_dict())
    
    # Get all linear layers
    layers = [module for module in new_model.modules() if isinstance(module, nn.Linear)]

    # Iterate over all layers
    with torch.no_grad():
        for i, layer in enumerate(layers):
            W = layer.weight
            b = layer.bias
            
            # Fist layer
            if i == 0:
                # Multiply D from the left
                new_W = (D[i] @ W)
                if b is not None:
                    new_b = (D[i] @ b.unsqueeze(1)).squeeze(1)
            # Middle layer
            elif i != len(layers) - 1:
                # Multiply D from the left and D^-1 from the right
                new_W = (D[i] @ (W @ D_inv[i-1]))
                if b is not None:
                    new_b = (D[i] @ b.unsqueeze(1)).squeeze(1)
            # Last layer
            else:
                # Multiply D^-1 from the right
                new_W = (W @ D_inv[i-1])
                new_b = b
            
            # Save new weights / biases
            layer.weight.copy_(new_W)


            # Scale gradients
            # Updated weight scaling with gradient hook to recover the original gradient scale.
            if i == 0:
                # Undo left scaling: multiply gradient from left by inv(D[i])^T.
                layer.weight.register_hook(lambda grad, l=i: grad)
            elif i != len(layers) - 1:
                # Undo left scaling by inv(D[i])^T and right scaling by (D_inv[i-1])^-T (which equals D[i-1]^T)
                layer.weight.register_hook(lambda grad, l=i: grad @ D[l-1].T)
            else:
                # Undo right scaling: multiply gradient from right by (D_inv[i-1])^-T.
                layer.weight.register_hook(lambda grad, l=i: grad @ D[l-1].T)



            if b is not None:
                layer.bias.copy_(new_b)
    
    return new_model




# == For testing, this file can be run standalone:

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(8, 3)
        self.fc2 = nn.Linear(3, 5)
        self.fc3 = nn.Linear(5, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def main():

    # Create model and reparametrize
    model = SimpleModel()
    start = time.time()
    new_model = reparametrize(model)
    end = time.time()
    print(f"Reparametrization took {end - start} seconds")


    print("=== Reparameterized Model Weights: ===")
    for name, param in new_model.named_parameters():
        print(name, param)

    print("\n\n")

    # Compute difference between original and reparametrized version
    differences = []

    # Compute input size of network
    input_size = 0
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            input_size = layer.in_features
    
    # Run multiple times to get average
    for _ in range(100):
        test_input = torch.randn(1, input_size)
        original_output = model(test_input)
        reparam_output = new_model(test_input)
        differences.append(torch.norm(original_output - reparam_output).item())
    
    print("=== Output Differences Over 100 Runs ===")
    print(sum(differences) / len(differences))

if __name__ == "__main__":
    main()

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
import time
import numpy as np
from torch import nn, optim
from utils.Training import train_model
from models.MnistNet import MnistNet
from models.GModel import GModel
from parameterSpaces.GSpace import GSpace
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy


"""
This file tests some properties of the GSpace

    - Check whether transforming forward and backward yields the same parameters as before
    - Check if number of Paths = Number of parameters - |hidden nodes|
    - Check if skeleton does in fact stay fixed during training
    - Check running time of forward and backward transformation
"""


def check_forward_backward(model, device):
    """
    This tests aims to proove that f(f^-1(θ)) = θ
    """
    # Deep copy the original parameters
    original_params = [p.clone().detach() for p in model.parameters()]

    gSpace = GSpace([p for p in model.parameters()], device)

    # Compute the G-space of the model
    start_time = time.time()
    g_space = gSpace.create_space()
    forward_time = time.time() - start_time

    # map back to the weight space
    start_time = time.time()
    gSpace.backward_projection(g_space)
    backward_time = time.time() - start_time

    # Print computing performance
    print(f"Forward mapping took {forward_time} s and backward mapping took {backward_time} s")

    # Compare the updated parameters with the original ones
    are_same = True
    for original, updated in zip(original_params, model.parameters()):
        if not torch.allclose(original, updated, atol=1e-6):
            return False

    print("Testing if forward -> backward maps to the same parameter values")
    self.assertTrue(areSame, "Forward and backward projection did not return the original parameters")



def test_parameters(model, device):
    """
    Tests that the parameter space of the GModel is by exactly the number of hidden nodes smaller
    """
    gmodel = GModel(model)  # Wrap the model with GModel

    # Count total number of parameters in the original model
    original_param_count = sum(p.numel() for p in model.parameters())
    # Count total number of parameters in the gmodel
    gmodel_param_count = sum(p.numel() for p in gmodel.parameters())

    parameter_difference = original_param_count - gmodel_param_count

    # Compute number of hidden nodes
    number_hidden_nodes = 0
    linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    if len(linear_layers) > 1:
        for layer in linear_layers[:-1]:  # exclude last linear layer (typically the output layer)
            number_hidden_nodes += layer.out_features

    print(f"Number of hidden nodes: {number_hidden_nodes}")

    assert number_hidden_nodes == parameter_difference, (
        f"Test failed: The dimensionality of GModel ({gmodel_param_count}) is not reduced by the number of hidden nodes: {number_hidden_nodes}"
    )
    print("Test passed: The GSpace is by the number of hidden nodes smaller")



def check_skeleton(model, device):
    """
    Check, if the skeleton weights stay the same during training
    """

    # Setup MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    batch_size = 64
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    val_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    gmodel = GModel(model)

    # Use G-SGD in order to keep skeleton weights
    optimizer = optim.SGD(gmodel.parameters(), lr=0.05)

    # Deep copy the original model before training
    original_model = copy.deepcopy(gmodel.model)

    # Train the model
    train_model(gmodel, train_loader, val_loader, optimizer, device, 2)

    # Starting in the second layer, check that every weight where the mask is 1
    # remains unchanged between the original model and the updated model.
    for layer_idx, (updated_param, orig_param) in enumerate(zip(gmodel.model.parameters(), original_model.parameters())):
        if layer_idx == 0:
            continue  # Skip the first layer
        this_mask = gmodel.g_space.mask[layer_idx]
        if not torch.equal(updated_param[this_mask == 1], orig_param[this_mask == 1]):
            raise AssertionError(f"Skeleton weights in layer {layer_idx} changed during training.")
    print("Test passed: Skeleton weights remain unchanged during training.")



if __name__ == "__main__":

    # Setup model
    model = MnistNet()
    device = "cpu"

    # Run tests
    check_forward_backward(model, device)
    test_parameters(model, device)
    check_skeleton(model, device)

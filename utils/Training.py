from torch import nn, optim
from tqdm import tqdm
import os
import torch
import time


# training Function
def train_model(model, train_loader, val_loader, optimizer, device, epochs):
    """
    Train the given model using Cross-Entropy-Loss. Return the test and validation losses
    """

    criterion = nn.CrossEntropyLoss()
    model.train()

    train_losses, val_losses, epoch_times = [], [], []

    # Compute initial loss
    initial_val_loss = validate_model(model, val_loader, criterion, device)
    train_losses.append(initial_val_loss)
    val_losses.append(initial_val_loss)


    for epoch in range(epochs):
        # Measure performance (time per epoch)
        start_time = time.time()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):

            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Run model and compute loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # G-Model: Update the G-space with the new gradients
            if model.__class__.__name__ == "GModel":
                model.update_gradients()

            # Run optimizer step and compute duration
            start_time = time.time()
            optimizer.step()
            forward_time = time.time() - start_time

            running_loss += loss.item()
        
        # Store losses
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
        epoch_times.append(time.time() - start_time) # compute duration of one epoch and save it in array
    

    return train_losses, val_losses, epoch_times


def validate_model(model, val_loader, criterion, device):
    """
    Compute validation losses
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss



def analyze_model_weights(model, optimizer=None):
    """
    ==== Debugging function ====
    Can help detect numerical instabilities.

    Prints:
    - The weight closest to zero
    - The largest and smallest weight values
    - The largest learning rate (if an optimizer is provided)
    
    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
        optimizer (torch.optim.Optimizer, optional): The optimizer to retrieve learning rates.
    """
    min_abs_weight = None
    min_abs_value = float("inf")
    max_weight = float("-inf")
    min_weight = float("inf")

    # Iterate through model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Find min and max weight values
            max_weight = max(max_weight, param.max().item())
            min_weight = min(min_weight, param.min().item())

            # Find the weight closest to zero
            abs_param = param.abs()
            min_val = abs_param.min().item()
            if min_val < min_abs_value:
                min_abs_value = min_val
                min_abs_weight = param.view(-1)[torch.argmin(abs_param)].item()  # Get actual value

    print(f"Weight closest to 0: {min_abs_weight}")
    print(f"Largest weight value: {max_weight}")
    print(f"Smallest weight value: {min_weight}")

    # Find the largest learning rate (if optimizer is provided)
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            max_grad = max(max_grad, param.grad.abs().max().item())
    print(f"Largest gradient: {max_grad}")

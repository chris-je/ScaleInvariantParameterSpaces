from torch import nn, optim
from optimizers.gsgd import GSGD

# select an optimizer according to the arguments and return it
def get_optimizer(model, optimizer_name, learning_rate, device):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "gsgd":
        return GSGD(list(model.parameters()), lr=learning_rate, device=device)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

from torch import nn, optim
from optimizers.GOptimizer import GOptimizer
from parameterSpaces.FixModel import FixModel

def get_optimizer(model, optimizer_name, learning_rate, device):
    """
    returns the specified optimizer. Supports the use of G-space and the weight fixing method
    """

    # Fix weight to 1 for weight fixing?
    fix_weights = True


    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate)
    # G-space based
    elif optimizer_name.lower() == "gsgd":
        return GOptimizer(list(model.parameters()), optimizer_class=optim.SGD,  lr=learning_rate)
    elif optimizer_name.lower() == "gadam":
        return GOptimizer(list(model.parameters()), optimizer_class=optim.Adam, lr=learning_rate)
    # Weight fixing based (will update the model as well)
    elif optimizer_name.lower() == "fsgd":
        FixModel(model, device, fix_weights)
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "fadam":
        FixModel(model, device, fix_weights)
        return optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

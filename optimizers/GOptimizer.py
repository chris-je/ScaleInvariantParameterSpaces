import torch
import torch.nn as nn
from torch.optim import Optimizer
from parameterSpaces.GSpace import GSpace



class GOptimizer(Optimizer):
    """
    Encapsulate an optimizer and let it train on the G-Space instead of the network parameters directly.
    This class behaves like a normal optimizer class.
    """
    
    def __init__(self, params, optimizer_class, lr):

        # Get device from model parameters
        device = params[0].device

        defaults = dict(lr=lr, device=device)
        super(GOptimizer, self).__init__(params, defaults)

        self.lr = lr  # Initial learning rate
        self.device = device

        # External optimizer class for g_space
        if optimizer_class:
            self.optimizer_class = optimizer_class
        else:
            self.optimizer_class = None

        # create whole G-Space
        self.g_space = GSpace(params, device)
    

    def step(self, closure=None):
        """
        Create the G-space and do a single optimizer step on it
        """
        self.g_space.forward_projection()
        self.update_values(self.g_space.g_space)
        self.g_space.backward_projection()


    def update_values(self, g_space):
        """
        Runs a single optimization step using the provided optimizer class
        """

        if g_space.grad is None:
            raise ValueError("Gradients for g_space are not available.")

        # Use external optimizer if provided
        if self.optimizer_class:
            optimizer = self.optimizer_class([g_space], lr=self.lr)
            optimizer.step()
        else:
            # Use default SGD for update
            with torch.no_grad():
                g_space -= self.lr * g_space.grad

        # Zero out the gradients after the update
        g_space.grad.zero_()
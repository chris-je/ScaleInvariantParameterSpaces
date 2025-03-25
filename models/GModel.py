import torch
from torch import nn, optim
from parameterSpaces.GSpace import GSpace

class GModel(nn.Module):
    """
    Encapsulate a pytorch model.
    Intercepts the parameters() method to return the G-space, otherwise redirects function calls to the model
    """

    def __init__(self, model, additional_params=None):
        super().__init__()
        
        self.model = model
        self.device = next(self.model.parameters()).device

        # Create G-Space
        params = list(model.parameters())
        self.g_space = GSpace(params, self.device)
        self.g_space.create_space()


    def forward(self, *args, **kwargs):
        # Map from G-space to the model
        self.g_space.backward_projection()

        # Zero out gradients because the optimizer won't have access to the original model
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        return self.model(*args, **kwargs)


    # Map gradients from weight to G-space
    def update_gradients(self):
        self.g_space.update_gradients()


    # Redirect functions to the model, if not implemented here
    def __getattr__(self, name):
        if name != 'model' and hasattr(self, 'model') and hasattr(self.model, name):
            print(f"function redirected: {name}")
            return getattr(self.model, name)
        return super().__getattr__(name)
    

    # Return G-space as parameters
    def parameters(self, recurse=True):
        return [self.g_space.g_space]
    
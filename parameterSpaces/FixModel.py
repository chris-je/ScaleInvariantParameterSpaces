import torch
from torch import nn


class FixModel:
    """
    Selects the fixed weights and set their gradient to 0. Optionally, set them to a specific value
    """

    def __init__(self, model, device, modify_skeleton):
        self.device = device
        self.model = model
        selection_factor = 1

        # Create initial G-Space
        masks = self.select_fixed_weights(model)
        self.fix_model(model, masks, modify_skeleton, selection_factor)

    def updated_model(self):
        return self.model



    def select_fixed_weights(self, model):
        """
        Generates a mask, which will be used to determine skeleton weights.
        """
        masks = []
        layers = self.extract_invariant_layers(model)  # Extract layers with ReLU activations

        for idx, layer in enumerate(layers):

            # === is Linear layer
            if isinstance(layer, nn.Linear):
                out_features, in_features = layer.weight.shape
                mask = torch.zeros_like(layer.weight)
                
                if out_features == in_features:
                    mask.fill_diagonal_(1)
                elif out_features > in_features:
                    # More outputs than inputs: only the first in_features diagonal entries
                    for j in range(in_features):
                        mask[j, j] = 1
                else:
                    # Wider matrix: repeat the diagonal pattern to the right.
                    for row in range(out_features):
                        col = row
                        while col < in_features:
                            mask[row, col] = 1
                            col += out_features
                masks.append(mask)

            # === is convolutional layer
            elif isinstance(layer, nn.Conv2d):
                # Get the shape of the convolutional layers weights
                weight_shape = layer.weight.shape  # (out_channels, in_channels, kernel_h, kernel_w)
                out_channels, in_channels, kernel_h, kernel_w = weight_shape

                # Create a mask tensor with the same shape
                mask = torch.zeros(weight_shape, dtype=torch.float32, device=layer.weight.device)

                if out_channels == in_channels:
                    # Equal number of input and output channels
                    for i in range(out_channels):  
                        mask[i, i, 0, 0] = 1  # Set weight at (i, i, 0, 0)

                elif in_channels < out_channels:
                    # More output channels than input channels
                    for i in range(in_channels): 
                        mask[i, i, 0, 0] = 1  # Set weight at (i, i, 0, 0)
                    # No action for extra output channels (out_channels > in_channels)

                else:
                    # More input channels than output channels
                    for i in range(in_channels):
                        mask[i % out_channels, i, 0, 0] = 1  # Cycle through output channels

                masks.append(mask)  # Store the mask for this Conv2d layer

        return masks



    def fix_model(self, model, masks, modify_skeleton, selection_factor=1):
        """
        Registers backward hooks on weights selected by the masks
        to zero out gradients.
        Can also set the fixed weights to a specific number.
        """
        # Extract invariant layers
        layers = self.extract_invariant_layers(model)

        # Iterate over all layers
        for idx, layer in enumerate(layers):
            current_mask = masks[idx]

            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                # Set weights to selection factor
                if modify_skeleton:
                    layer.weight.data[current_mask == 1] = selection_factor
                
                # Set the gradient to 0, where mask==1
                layer.weight.register_hook(lambda grad, m=current_mask: grad * (1 - m))



    def extract_invariant_layers(self, model):
        """
        Select layers, if pattern: Layer -> ReLU -> Layer is found
        """

        layers = []
        previous_layer = None  # Store the last seen Linear or Conv2d layer
        previous_activation = None  # Track the activation before the current layer

        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                previous_activation = module  # Track ReLU/LeakyReLU activation

            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                # Check if the previous layer is of the same type and activation is ReLU, LeakyReLU, or None
                if previous_layer and isinstance(previous_layer, type(module)) and (
                    previous_activation is None or isinstance(previous_activation, (nn.ReLU, nn.LeakyReLU))
                ):
                    layers.append(module)

                # Update tracking variables
                previous_layer = module
                previous_activation = None  # Reset activation tracking for next iteration

        return layers


    
    # TODO: check, experimental. Create a parameter vector
    # def create_vector_space(self, model, mask):
    #     param_vector = utils.parameters_to_vector(model.parameters())

# class ParameterVectorSpace:
    # def __init__(self, model, masks):
    #     self.model = model
    #     self.masks = masks
    #     self.params, self.shapes, self.indices = self._collect_params()
    #     self.vector = torch.nn.Parameter(self.params)

    # def _collect_params(self):
    #     params, shapes, indices = [], [], []
    #     for param, mask in zip(self.model.parameters(), self.masks):
    #         assert param.shape == mask.shape, "Mask shape must match parameter shape."
    #         shapes.append(param.shape)

    #         flat_param = param.view(-1)
    #         flat_mask = mask.view(-1)

    #         selected_indices = (flat_mask == 0).nonzero(as_tuple=True)[0]
    #         indices.append(selected_indices)

    #         selected_params = flat_param[selected_indices]
    #         params.append(selected_params)

    #     return torch.cat(params), shapes, indices

    # def update_model_params(self):
    #     idx = 0
    #     for param, shape, selected_indices in zip(self.model.parameters(), self.shapes, self.indices):
    #         flat_param = param.view(-1)
    #         num_selected = len(selected_indices)

    #         flat_param[selected_indices] = self.vector[idx:idx + num_selected]
    #         idx += num_selected


import torch
import numpy as np



class GSpace:
    """
    Converts a set of pytorch parameters into scale-invariant G-Space and back

    Attributes:
        forward_projection: Creates the G-space and adds gradients
        create_space: Create the G-space without gradients
        update_gradients: Updates the gradients of an existing G-space
        backward_projection: Maps back from G-space to weight space
    """

    def __init__(self, params, device):
        self.device = device
        self.params = params
        self.g_space = None

        # Categorize the weights into skeleton and non-skeleton
        self.select_skeleton_weights()
        # Only needed when skeleton is not set to 1: compute the path scaling factors
        self.compute_path_scaling()
        # Initialize G-space
        self.create_space()


    def forward_projection(self):
        """
        Creates a new G-space with gradients
        """
        self.create_space()
        self.update_gradients()

        return self.g_space



    def create_space(self):
        """
        Transforms the parameters into G-Space. Doesn't transform the gradients, call update_gradients() for them.
        """
        v_h = torch.empty(0).to(self.device)
        v_i = torch.empty(0).to(self.device)
        v_j = torch.empty(0).to(self.device)

        # We need w_j for further computation
        w_j = torch.empty(0).to(self.device)

        param_count = 0

        # iterate over all layers
        for layer_idx, layer in enumerate(self.params):
            this_mask = self.mask[layer_idx]

            param_count += layer.data.numel()

            if layer_idx == 0:
                v_j = layer.data[this_mask == 1]
                v_h = layer.data[this_mask == 0]

                # todo: unsqueeze stimmt?
                w_j = layer.data[this_mask == 1]

                # s_diag = self.s[1].diag().detach()  # shape: (n,)
                # v_j = (layer.data * s_diag.unsqueeze(1))[this_mask == 1]
                # v_h = (layer.data * s_diag.unsqueeze(1))[this_mask == 0]
                # print(f"s at layer 0: {self.s[layer_idx]}")
            else:
                v_i_all = layer.data * w_j
                v_i = torch.cat((v_i, v_i_all[this_mask == 0].view(-1)))


                # v_i_all = layer.data * w_j *  self.s[layer_idx].detach()
                # print(f"layer create {layer_idx}:")
                # print(v_i_all[this_mask == 0].view(-1))


        # Compute bias vector
        # biases = []
        # for layer in self.params:
        #     if layer.bias is not None:
        #         biases.append(layer.bias.view(-1))
        # bias_space = torch.cat(biases) if biases else torch.empty(0, device=self.device)

        v_j_new = v_j.view(-1)
        v_h_new = v_h.view(-1)

        # Put G-space together
        g_space = torch.cat((v_j_new, v_h_new, v_i))

        # Save size of subspaces (improves performance significantly)
        self.v_j_size = v_j.size(0)
        self.v_h_size = v_h.size(0)

        # Statistics
        # g_parameters = g_space.shape[0]
        # g_space_difference = param_count - g_parameters
        # g_space_percent = g_space_difference / g_parameters * 100
        # print(f"G-Space is {g_space_percent}% smaller: {g_parameters} parameters")

        # Save w_j to allow correct backprojection
        self.w_j = w_j

        self.g_space = g_space
        return g_space


    # Apply gradients to the G-Space
    def update_gradients(self, g_space = None):
        """
        Transforms gradients from parameter space into G-Space.
        Assumes that the weights in weight space have not changed since the last projection.
        If that's the case, call create_space() before.
        """
        
        if g_space is None:
            g_space = self.g_space

        # Psi is a constant we need for the computation of the gradient of v_j
        self.compute_psi()

        v_j = torch.empty(0).to(self.device)

        grad_v_j = torch.empty(0).to(self.device)
        grad_v_h = torch.empty(0).to(self.device)
        grad_v_i = torch.empty(0).to(self.device)

        # iterate over all layers
        for layer_idx, layer in enumerate(self.params):
            this_mask = self.mask[layer_idx]

            if layer_idx == 0:
                # s_diag = self.s[1].diag().detach()
                v_j = layer.data[this_mask == 1]
                grad_v_j = layer.grad[this_mask == 1] - self.psi * (1 / v_j)
                grad_v_h = layer.grad[this_mask == 0]
                # Conbstant path scale:
                # grad_v_j = (layer.grad / s_diag.unsqueeze(1))[this_mask == 1] - self.psi * (1 / v_j)
                # grad_v_h = (layer.grad / s_diag.unsqueeze(1))[this_mask == 0]
            else:
                # Constant path scale
                # grad_v_i = torch.cat((grad_v_i, (layer.grad.data / (v_j * self.s[layer_idx].detach()))[this_mask == 0].view(-1)))
                grad_v_i = torch.cat((grad_v_i, (layer.grad.data / v_j.view(1, -1))[this_mask == 0].view(-1)))


        # Compute bias vector
        # biases = []
        # for layer in self.params:
        #     if layer.bias is not None:
        #         biases.append(layer.bias.view(-1))
        
        # if biases:
        #     bias_space = torch.cat(biases, dim=0)
        #     bias_space.requires_grad = True
        # else:
        #     bias_space = torch.empty(0, device=self.device, requires_grad=True)

        # fit g space together
        g_space_grad = torch.cat((grad_v_j, grad_v_h, grad_v_i))
        g_space.grad = g_space_grad

        self.g_space = g_space

        return g_space


    def compute_psi(self):
        """
        Compute psi (ψ)
        Psi is a factor needed for the computation of the gradient of v_j
        """
        first_layer_size = self.params[0].shape[0]
        psi = torch.zeros(first_layer_size).to(self.device)

        # iterate through all layers to calculate skeleton and non-skeleton gradients / path values?
        for layer_idx, layer in enumerate(self.params):
            this_mask = self.mask[layer_idx]
            if layer_idx != 0:
                # handle non-skeleton weights
                non_skeleton_weights = layer.data * (1 - this_mask)
                non_skeleton_gradients = layer.grad.data * (1 - this_mask)

                # accumulate dw_i * w_i for layers > 0:
                psi[:non_skeleton_weights.shape[1]] += (non_skeleton_weights * non_skeleton_gradients).sum(0)  # Sum over input
        self.psi = psi
        return psi

    

    def backward_projection(self, g_space = None):
        """
        Transform the gradients back to weight space
        """
        if g_space is None:
            g_space = self.g_space
        
        # Decompose g_space into v_j, v_h, and v_i
        v_j_size = self.v_j_size
        v_h_size = self.v_h_size

        v_j = g_space[:v_j_size]
        v_h = g_space[v_j_size:v_j_size + v_h_size]
        v_i = g_space[v_j_size + v_h_size:]

        # If w_k are not set to one, we need to retrieve w_j
        w_j = v_j
        if self.w_j is not None:
            w_j = self.w_j
        w_j_new = w_j

        #bias_size = sum(layer.bias.numel() for layer in self.params if layer.bias is not None)

        # v_j = g_space[:v_j_size]
        # v_h = g_space[v_j_size:v_j_size + v_h_size]
        # v_i = g_space[v_j_size + v_h_size:-bias_size] if bias_size > 0 else g_space[v_j_size + v_h_size:]
        #biases = g_space[-bias_size:] if bias_size > 0 else torch.empty(0, device=self.device)


        # Reconstruct layers from G-space
        v_i_idx = 0
        #bias_idx = 0

        # Iterate over all layers
        for layer_idx, layer in enumerate(self.params):
            this_mask = self.mask[layer_idx]

            if layer_idx == 0:
                # Update first layer with v_j and v_h
                layer.data[this_mask == 1] = v_j
                layer.data[this_mask == 0] = v_h
                # Constant scale update:
                # print(f"Layer data before: {layer.data}")
                # layer.data = layer.data / self.s[1].detach().diag().unsqueeze(1)
                w_j_new = layer.data[this_mask == 1]
                # print(f"Layer data after: {layer.data}")
                # print("==================")
            else:
                # Reconstruct full matrix for v_i
                layer_shape = layer.data.shape
                v_i_layer = torch.zeros(layer_shape, device=self.device)

                # v_i_values = torch.zeros(layer_shape, device=self.device)
                # Count v_i in layer
                mask_sum = (this_mask == 0).sum()
                # Map v_i back to w_i
                v_i_layer[this_mask == 0] = v_i[v_i_idx:v_i_idx + mask_sum]
                layer.data[this_mask == 0] = (v_i_layer / w_j_new)[this_mask == 0]
                # Constant path scale:
                # layer.data[this_mask == 0] = (v_i_layer / (w_j_new * self.s[layer_idx].detach()))[this_mask == 0]

                # Constant path factor
                # print(f"Layer data before:\n{layer.data[this_mask == 0]}")
                # layer.data[this_mask == 0] = (layer.data / self.s[layer_idx].detach())[this_mask == 0]
                # print(f"Layer data after:\n{layer.data[this_mask == 0]}")
                # print("====================")
                # before = layer.data.clone()
                # layer.data[this_mask == 0] = (layer.data / self.s[layer_idx].detach())[this_mask == 0]
                # after = layer.data

                # diff = (after - before)[this_mask == 0]
                # print("Nonzero differences:\n", diff[diff != 0])



                # Update weights where mask == 0
                v_i_idx += mask_sum

            # Restore biases
            # if layer.bias is not None:
            #     num_bias = layer.bias.numel()
            #     layer.bias.data = biases[bias_idx:bias_idx + num_bias]
            #     bias_idx += num_bias
            

    def compute_path_scaling(self):
        """
        Compute the path scale matrix. It will contain the product of all w_k along the path
        It will contain constant factors needed for the computation of the path value
        in the case that our skeleton weights wk are not set to 1.
        """

        num_layers = len(self.params)
        # Initialize lists for path scaling matrices
        # We ignore layer 0
        self.s_forward = [None] * num_layers
        self.s_backward = [None] * num_layers
        self.s = [None] * num_layers

        # Compute s_forward for layers 1 .. num_layers-1
        for layer_idx in range(1, num_layers):
            if layer_idx == 1:
                # For the first computed layer, clone the weights
                self.s_forward[layer_idx] = self.params[layer_idx].clone()
            elif layer_idx == num_layers - 1:
                # For the last layer, check if the matrix is wide
                last_param = self.params[layer_idx].clone()
                r, c = last_param.shape
                last_layer_sq = last_param
                if r < c:
                    # If wide, tile the matrix along the rows to form a square matrix of size (c, c)
                    #todo use other implementation
                    rep = int(np.ceil(c / r))
                    tiled = np.tile(last_param.detach().cpu().numpy(), (rep, 1))
                    tiled = torch.tensor(tiled, dtype=last_param.dtype, device=last_param.device)
                    last_layer_sq = tiled[:c, :c]
                # Add the current layer's weight elementwise to s_forward from the previous layer
                self.s_forward[layer_idx] = last_layer_sq * self.s_forward[layer_idx - 1]
            else:
                # Add the current layer's weight elementwise to s_forward from the previous layer
                self.s_forward[layer_idx] = self.params[layer_idx] * self.s_forward[layer_idx - 1]

        # Compute s_backward for layers num_layers-1 down to 1.
        for layer_idx in range(num_layers - 1, 0, -1):
            if layer_idx == num_layers - 1:
                # For the last layer, check if the matrix is wide (more inputs than outputs)
                last_param = self.params[layer_idx].clone()
                r, c = last_param.shape
                if r < c:
                    # If wide, tile the matrix along the rows to form a square matrix of size (c, c)
                    rep = int(np.ceil(c / r))
                    tiled = np.tile(last_param.detach().cpu().numpy(), (rep, 1))
                    tiled = torch.tensor(tiled, dtype=last_param.dtype, device=last_param.device)
                    self.s_backward[layer_idx] = tiled[:c, :c]
                else:
                    self.s_backward[layer_idx] = last_param
            else:
                # Add the current layer's weight elementwise to s_backward from the next layer
                self.s_backward[layer_idx] = self.params[layer_idx] * self.s_backward[layer_idx + 1]

        # Compute s for layers that have both a previous and a next layer
        # For each element (m, n) in the layer's weight matrix:
        # s[m, n] = (s_forward from one layer earlier at index (n, n)) +
        #           (s_backward from one layer later at index (m, m))
        # We compute s for layers 2 to num_layers-2.
        for layer_idx in range(2, num_layers - 1):
            weight = self.params[layer_idx]
            s_mat = torch.zeros_like(weight)
            rows, cols = weight.shape
            for m in range(rows):
                for n in range(cols):
                    s_mat[m, n] = self.s_forward[layer_idx - 1][n, n] * self.s_backward[layer_idx + 1][m, m]
            self.s[layer_idx] = s_mat


            # print(f"forward of layer {layer_idx}:")
            # print(self.s_forward[layer_idx - 1][n, n])
            # print(f"==: {s_mat}")

            # # Boundary cases: compute s when only one neighbor scaling is defined
            # if num_layers > 1:
            #     # For the first parameter layer (index 1), use the backward scaling
            #     for m in range(rows):
            #         for n in range(cols):
            #             self.s[1, m, n] = self.s_forward[1][n, n]
            #     # Compute for first layer, needs to be extended
            #     #todo
            #     # For the last parameter layer (index num_layers - 1), use the forward scaling
            #     for m in range(rows):
            #         for n in range(cols):
            #             self.s[1] = self.s_forward[layer_idx - 1][n, n]
            #     self.s[num_layers - 1] = self.s_forward[num_layers - 1][: self.params[num_layers - 1].shape[0], : self.params[num_layers - 1].shape[1]]

        # Set 0-th layer
        rows, cols = self.params[1].shape
        s_first = torch.zeros_like(self.params[1])
        for m in range(rows):
            s_first[m, :] = self.s_backward[1][m, m]
        self.s[0] = s_first

        rows, cols = self.params[1].shape
        s_mat_fir = torch.zeros_like(weight)
        for m in range(rows):
            for n in range(cols):
                s_mat_fir[m, n] = self.s_backward[2][m, m]
        self.s[1] = s_mat_fir

        # Set last layer
        # For the last parameter layer, use the forward scaling.
        # Fill each column with s_forward[num_layers - 1][n, n]
        rows, cols = self.params[num_layers - 1].shape
        s_last = torch.zeros_like(self.params[num_layers - 1])
        for n in range(cols):
            s_last[:, n] = self.s_forward[num_layers - 2][n, n]
        self.s[num_layers - 1] = s_last
        # print(self.s)
        # print("========== this was s")




    def select_skeleton_weights(self):
        """
        Iterate over each layer and accumulate mask matrices. mask==1 means, that a weight is a skeleton weight.
        Set the skeleton weights to a constant
        """

        num_layer = len(self.params)
        num_hidden = num_layer - 1
        mask = []

        # Generate masks for each layer
        for layer_idx, layer in enumerate(self.params):
            out_shape, in_shape = layer.shape

            # Use the select_skeleton_weight function to create the mask
            mask_layer = self.select_skeleton_weight(layer_idx, out_shape, in_shape).to(self.device)
            mask.append(mask_layer)

            # Set skeleton to constant number
            custom_value = 1
            layer.data = layer.data * (1 - mask_layer) + custom_value * mask_layer

        self.mask = mask

        return mask


    def select_skeleton_weight(self, layer_idx, rows, columns) -> torch.Tensor:
        """
        Selects the skeleton weights based on the skeleton method
        """

        ratio = max(rows, columns) // min(rows, columns) + 1
        eye_matrix = torch.eye(min(rows, columns))

        if rows > columns:
            mask = eye_matrix.repeat(ratio, 1)[:rows, :]
        elif layer_idx == 0:
            eye_matrix = torch.eye(max(rows, columns))
            mask = eye_matrix[:rows, :columns]
        else:
            mask = eye_matrix.repeat(ratio, ratio)[:rows, :columns]

        return mask.to(self.device)

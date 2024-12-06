import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Optimizer
import torch.nn.parallel
import numpy as np
import scipy.sparse as sparse
import torchvision.models
import random
import math


class GSGD(Optimizer):
    
    # initialization function of the gSGD optimizer
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, device="cpu"):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, device=device)
        super(GSGD, self).__init__(params, defaults)
        self.lr = lr  # Initial learning rate
        self.device = device

        # TODO: implement momentum and dampening
        self.params = params
        # calculate masks to differentiate between skeleton and non-skeleton weights
        self.mask, self.s_mask_idx, self.s_mask_idx_shape = self.generate_skeleton_masks()


    # Clears the gradients of all optimized tensors
    def zero_grad(self):
        # todo: use super.zero_grad() instead
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


    # calculate masks to identify skeleton weights in the model
    def generate_skeleton_masks(self):
        num_layer = len(self.params)
        num_hidden = num_layer - 1
        mask = []

        # Use first layer as skeleton layer
        self.skeleton_layer_index = 0
        self.skeleton_layer_size = self.params[0].shape[0]  # Size of the skeleton layer

        # generate masks for each layer
        for layer_idx, layer in enumerate(self.params):
            outcome, income = layer.shape
            loc = False
            if layer_idx == 0:
                loc = 'first'
            mask.append(self.generate_layer_mask(outcome, income, loc).to(self.device))
            # Apply mask to layer data to separate skeleton weights
            layer.data = layer.data * (1 - mask[-1]) + mask[-1]
        # get the non-zero indices of the skeleton mask
        s_mask_idx = (mask[self.skeleton_layer_index] != 0).nonzero().transpose(0, 1)
        s_mask_idx_shape = mask[self.skeleton_layer_index].shape

        return (mask, s_mask_idx, s_mask_idx_shape)

    # TODO: check if this aligns with the paper (this is just a refactored version)
    def generate_layer_mask(self, out_shape, in_shape, loc='middle'):
        # generate mask for first layer
        ratio = max(out_shape, in_shape) // min(out_shape, in_shape) + 1
        eye_matrix = torch.eye(min(out_shape, in_shape))
        
        if out_shape > in_shape:
            mask = eye_matrix.repeat(ratio, 1)[:out_shape, :]
        elif loc == "first":
            eye_matrix = torch.eye(max(out_shape, in_shape))
            mask = eye_matrix[:out_shape, :in_shape]
        else:
            mask = eye_matrix.repeat(ratio, ratio)[:out_shape, :in_shape]
        
        return mask.to(self.device)

    # recover sparse skeleton layers
    def recover_s_layer(self, value, idx, shape):
        return torch.sparse.FloatTensor(idx, value, shape).to_dense().to(self.device)
        

    # calculate the path ratio R used for skeleton weight updates
    def compute_path_ratio(self, lr, skeleton_weights, skeleton_gradients, sigma_dw, v_value):
        return (1 - lr * (skeleton_gradients * skeleton_weights - sigma_dw) / (v_value * v_value))

    # Function to update model parameters using calculated path ratios
    def weight_allocation(self, mask, R, v_value):
        for layer_idx, layer in enumerate(self.params):
            this_mask = mask[layer_idx]

            if layer_idx == self.skeleton_layer_index:
                layer_is_skeleton = True
            else:
                layer_is_skeleton = False

            if layer_is_skeleton:
                # Update skeleton layer weights
                layer.data = (layer.data - self.lr * layer.grad.data) * (1 - this_mask) + \
                             layer.data * self.recover_s_layer(value=R,
                                                               idx=self.s_mask_idx,
                                                               shape=self.s_mask_idx_shape)

            elif layer_idx > self.skeleton_layer_index:
                # Update layers after the skeleton layer
                out_shape, in_shape = layer.data.shape
                layer.data = (layer.data - self.lr * layer.grad.data / (v_value[:layer.data.shape[1]].view(1, -1) ** 2)) / (R[:in_shape].view(1, -1)) * (1 - this_mask) + \
                    layer.data * this_mask

            else:
                print("Error in G-SGD: before skeleton layer?")

            # elif layer_idx < self.skeleton_layer_index:
            #     # Update layers before the skeleton layer
            #     out_shape, in_shape = layer.data.shape
            #     layer.data = (layer.data - self.lr * layer.grad.data / (v_value[:layer.data.shape[0]].view(-1, 1) ** 2)) / (R[:out_shape].view(-1, 1)) * (1 - this_mask) + \
            #         layer.data * this_mask


    # perform a single optimization step using g-sgd
    def step(self, closure=None):
        num_layer = len(self.params)
        mask = self.mask
        lr = self.lr

        # initialize lists for skeleton and non-skeleton weights
        skeleton_weights = []
        non_skeleton_weights = []
        skeleton_gradients = []
        non_skeleton_gradients = []
        # Initialize sigma * dw accumulator
        sigma_dw = torch.zeros(self.skeleton_layer_size).to(self.device)

        # iterate through all layers to calculate skeleton and non-skeleton gradients / path values?
        for layer_idx, layer in enumerate(self.params):
            this_mask = mask[layer_idx]
            if layer_idx != self.skeleton_layer_index:
                # handle non-skeleton weights
                non_skeleton_weights = layer.data * (1 - this_mask)
                non_skeleton_gradients = layer.grad.data * (1 - this_mask)

                # accumulate sigma * dw for layers before and after the skeleton layer
                if layer_idx < self.skeleton_layer_index:
                    sigma_dw[:non_skeleton_weights.shape[0]] += (non_skeleton_weights * non_skeleton_gradients).sum(1)  # Sum over output
                elif layer_idx > self.skeleton_layer_index:
                    sigma_dw[:non_skeleton_weights.shape[1]] += (non_skeleton_weights * non_skeleton_gradients).sum(0)  # Sum over input
            else:
                # handle skeleton weights
                v_value = (layer.data * this_mask).sum(1)
                skeleton_weights = v_value
                skeleton_gradients = (layer.grad.data * this_mask).sum(1)

        # compute path ratios
        path_ratio = self.compute_path_ratio(lr=lr,
            skeleton_weights=skeleton_weights,
            skeleton_gradients=skeleton_gradients,
            sigma_dw=sigma_dw,
            v_value=v_value
            )

        # update the model parameters using weight allocation
        self.weight_allocation(mask, path_ratio, v_value)
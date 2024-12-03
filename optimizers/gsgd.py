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
        self.lr_0 = self.lr_t = lr  # Initial learning rate
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
        self.skeleton_layer_index = 0
        num_layer = len(self.params)
        num_hidden = num_layer - 1
        mask = []

        # Determine the largest hidden layer to use as the skeleton layer
        largest_layer = [0, 0]
        for i in range(num_layer - 1):
            num_parameters = self.params[i].shape[0]
            if num_parameters > largest_layer[1]:
                largest_layer = [i, num_parameters]

        self.skeleton_layer_index = largest_layer[0]  # Index of the skeleton layer
        self.skeleton_layer_size = largest_layer[1]  # Size of the skeleton layer

        # generate masks for each layer
        for layer_idx, layer in enumerate(self.params):
            outcome, income = layer.shape
            if layer_idx == 0:
                loc = 'first'
            elif layer_idx == num_layer - 1:
                loc = 'last'
            else:
                loc = 'middle'
            mask.append(self.generate_layer_mask(outcome, income, loc).to(self.device))
            # Apply mask to layer data to separate skeleton weights
            layer.data = layer.data * (1 - mask[-1]) + mask[-1]
        # get the non-zero indices of the skeleton mask
        s_mask_idx = (mask[self.skeleton_layer_index] != 0).nonzero().transpose(0, 1)
        s_mask_idx_shape = mask[self.skeleton_layer_index].shape

        return (mask, s_mask_idx, s_mask_idx_shape)

    # recover sparse skeleton layers
    def recover_s_layer(self, value, idx, shape):
        return torch.sparse.FloatTensor(idx, value, shape).to_dense().to(self.device)

    # generate mask for skeleton weights for a given single layer
    def generate_layer_mask(self, out_shape, in_shape, loc='middle'):
        # generate mask for first layer
        if loc == 'first':
            ratio = out_shape // in_shape + 1
            out_idx = list(range(out_shape))
            in_idx = list(range(in_shape)) * ratio
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([x for x in idx_tmp]).transpose(0, 1)
            return(
                self.recover_s_layer(
                    idx=idx,
                    value=torch.ones(out_shape),
                    shape=[out_shape, in_shape])
            )

        # generate mask for last layer
        elif loc == 'last':
            ratio = in_shape // out_shape + 1
            out_idx = list(range(out_shape)) * ratio
            in_idx = list(range(in_shape))
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([x for x in idx_tmp]).transpose(0, 1)
            return(
                self.recover_s_layer(
                    idx=idx,
                    value=torch.ones(in_shape),
                    shape=[out_shape, in_shape])
            )

        # generate mask for middle layers
        elif loc == 'middle':
            if in_shape > out_shape:
                ratio = in_shape // out_shape + 1
                out_idx = list(range(out_shape)) * ratio
                in_idx = list(range(in_shape))
            else:
                ratio = out_shape // in_shape + 1
                out_idx = list(range(out_shape))
                in_idx = list(range(in_shape)) * ratio

            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([x for x in idx_tmp]).transpose(0, 1)
            return(
                self.recover_s_layer(
                    idx=idx,
                    value=torch.ones(max(out_shape, in_shape)),
                    shape=[out_shape, in_shape])
            )

    # calculate the path ratio R used for skeleton weight updates
    def compute_R(self, lr, skeleton_weights, skeleton_gradients, sigma_dw, v_value):
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
                layer.data = (layer.data - self.lr_t * layer.grad.data) * (1 - this_mask) + \
                             layer.data * self.recover_s_layer(value=R,
                                                               idx=self.s_mask_idx,
                                                               shape=self.s_mask_idx_shape)

            elif layer_idx > self.skeleton_layer_index:
                # Update layers after the skeleton layer
                out_shape, in_shape = layer.data.shape
                layer.data = (layer.data - self.lr_t * layer.grad.data / (v_value[:layer.data.shape[1]].view(1, -1) ** 2)) / (R[:in_shape].view(1, -1)) * (1 - this_mask) + \
                    layer.data * this_mask

            elif layer_idx < self.skeleton_layer_index:
                # Update layers before the skeleton layer
                out_shape, in_shape = layer.data.shape
                layer.data = (layer.data - self.lr_t * layer.grad.data / (v_value[:layer.data.shape[0]].view(-1, 1) ** 2)) / (R[:out_shape].view(-1, 1)) * (1 - this_mask) + \
                    layer.data * this_mask


    # perform a single optimization step using g-sgd
    def step(self, closure=None):
        num_layer = len(self.params)
        mask = self.mask
        lr = self.lr_t

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
        path_ratio = self.compute_R(lr=lr,
            skeleton_weights=skeleton_weights,
            skeleton_gradients=skeleton_gradients,
            sigma_dw=sigma_dw,
            v_value=v_value
            )

        # update the model parameters using weight allocation
        self.weight_allocation(mask, path_ratio, v_value)
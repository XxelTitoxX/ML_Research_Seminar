# Implementation of simple MLP

"""
In the context of CatFlow, the MLP should take as input a vector resulting from the concatenation over all dimensions of categorical distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(MLP, self).__init__()
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim, output_dim, hidden_dim must be positive integers.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        activation = nn.ReLU
        layers: list[nn.Module] = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # input -> hidden
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())

            # hidden -> hidden (num_layers - 2 times)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())

            # hidden -> output
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)
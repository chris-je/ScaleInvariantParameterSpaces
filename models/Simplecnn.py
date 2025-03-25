import torch
import torch.nn as nn
import torch.nn.functional as F


class Simplecnn(nn.Module):
    def __init__(self, bias = True):
        super(Simplecnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias = bias)
        self.conv2 = nn.Conv2d(6, 6, 5, bias = bias)
        self.conv3 = nn.Conv2d(6, 6, 5, bias = bias)
        self.fc1 = nn.Linear(6*16*16, 16, bias = bias)
        self.fc2 = nn.Linear(16, 16, bias = bias)
        self.fc3 = nn.Linear(16, 10, bias = bias)
        self.ec_layer = {
            'cnn' : [ [0,2,4] ],
            'mlp' : [ [6,8,10]  ],
        }


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 6*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
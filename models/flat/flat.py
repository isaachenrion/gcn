import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


class Flat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.target_names = config.target_names
        self.state_dim, self.hidden_dim, self.readout_dim = config.state_dim, config.hidden_dim, sum(t.dim for t in config.target_names)
        self.bn0 = nn.BatchNorm1d(self.state_dim)
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.readout_dim)
        self.activation = nn.ReLU()

    def forward(self, vertices):
        x = vertices.view(vertices.shape[0], -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.activation(self.fc2(x))
        x = self.bn2(x)
        x = self.activation(self.fc3(x))
        x = self.bn3(x)
        x = self.fc4(x)

        if self.mode == 'clf':
            x = F.sigmoid(x)
        return x

def make_flat(config):
    return Flat(config)
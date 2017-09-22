import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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

        self.target_mean = torch.zeros(1, self.readout_dim)
        self.target_var = torch.zeros(1, self.readout_dim)
        self.alpha = 0.999

    def forward(self, vertices, dads):
        x = torch.cat((vertices.view(vertices.size()[0], -1), dads.view(dads.size()[0], -1)), 1)
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
        #elif self.mode == 'reg':
        #    x = x * Variable(self.target_var) + Variable(self.target_mean)
        return x

    def record(self, targets):
        mean = torch.mean(targets.data, 0, keepdim=True)
        var = torch.var(targets.data, 0, keepdim=True)

        self.target_mean = self.alpha * self.target_mean + (1 - self.alpha) * mean
        self.target_var = self.alpha * self.target_var + (1 - self.alpha) * var

    def number_of_parameters(self):
        n = 0
        for p in self.parameters():
            n += np.prod([s for s in p.size()])
        return n


def make_flat(config):
    return Flat(config)

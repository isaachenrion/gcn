import torch
import torch.nn as nn
import torch.nn.functional as F

class Recurrent2D(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, readout_dim):
        super().__init__()
        self.recurrent = GRUCell2D(input_dim1, input_dim2, hidden_dim)

    def forward(self, x):
        pass

        # x is (B, d1, d2)


class GRUCell2D(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_input = nn.Linear(input_dim, 3 * hidden_dim)
        self.linear_hidden2 = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.linear_hidden1 = nn.Linear(hidden_dim, 3 * hidden_dim)

    def forward(self, x, h1, h2):
        r_h1, z_h1, n_h1 = self.linear_hidden1(h1).chunk(3, 1)
        r_h2, z_h2, n_h2 = self.linear_input2(h2).chunk(3, 1)
        r_x, z_x, n_x = self.linear_input1(x1).chunk(3, 1)

        r = F.sigmoid(r_h1 + r_x1 + r_x2)
        z = F.sigmoid(z_h1 + z_x1 + z_x2)
        n = F.tanh(n_x1 + n_x2 + r * n_h1)
        h = (1 - z) * n + z * h
        return h

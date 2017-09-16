from .mpnn import BaseMPNN
from .embedding import make_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)
        self.activation = F.tanh
        self.embedding = make_embedding(config.embedding)

    def forward(self, x, dads):
        h = self.embedding(x)
        for i in range(self.n_iters):
            h = self.message_passing(h, dads)
        out = self.readout(h)
        return out

    def message_passing(self, h, dads):
        message = self.message(h)
        h = self.activation(torch.matmul(dads, message))
        return h

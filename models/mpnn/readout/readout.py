import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import namedtuple

from .set2set import Set2Vec

ReadoutConfig = namedtuple(
        'ReadoutConfig', [
            'hidden_dim',
            'readout_hidden_dim',
            'mode',
            'target_dim',
        ]
)

class Readout(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classify = (self.config.mode == 'clf')
        self.hidden_dim = config.hidden_dim
        self.target_dim = config.target_dim
        self.readout_hidden_dim = config.readout_hidden_dim
        self.activation = nn.LeakyReLU


    def forward(self, G):
        pass


class DTNNReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.readout_hidden_dim),
                self.activation(),
                nn.BatchNorm1d(self.readout_hidden_dim),
                nn.Linear(self.readout_hidden_dim, self.target_dim),
                )
        self.net = net

    def forward(self, h):
        bs, gd, dd = (s for s in h.size())
        x = h.view(-1, dd)
        x = self.net(x)
        x = x.view(bs, gd, -1)
        x = x.sum(1)
        return x

class FullyConnectedReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.readout_hidden_dim),
                self.activation(),
                nn.BatchNorm1d(self.readout_hidden_dim),
                nn.Linear(self.readout_hidden_dim, self.target_dim),
                )
        self.net = net

    def forward(self, h):
        x = torch.mean(h, 1)
        x = self.net(x)
        return x

class SetReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        self.set2vec = Set2Vec(self.hidden_dim, self.target_dim, config.readout_hidden_dim)

    def forward(self, h):
        x = self.set2vec(h)
        return x


class VCNReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        self.module_list = nn.ModuleList()
        for target in self.target_names:
            self.module_list.append(nn.Linear(self.hidden_dim, target.dim))

    def forward(self, G):
        h_dict = {v: G.node[v]['hidden'] for v in G.nodes()}
        out = {}
        for i, target in enumerate(self.target_names):
            out[target.name] = self.module_list[i](h_dict[target.name])
        return out

class VertexReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.readout_hidden_dim),
                self.activation(),
                nn.BatchNorm2d(self.readout_hidden_dim),
                nn.Linear(self.readout_hidden_dim, self.target_dim),
                )
        self.net = net

    def forward(self, h):
        bs, gd, dd = (s for s in h.size())
        x = h.view(-1, dd)
        x = self.net(x)
        x = x.view(bs, gd, -1)
        return x


def make_readout(readout_config):
    if readout_config.function == 'fully_connected':
        return FullyConnectedReadout(readout_config.config)
    elif readout_config.function == 'dtnn':
        return DTNNReadout(readout_config.config)
    elif readout_config.function == 'vcn':
        return VCNReadout(readout_config.config)
    elif readout_config.function == 'vertex':
        return VertexReadout(readout_config.config)
    elif readout_config.function == 'set':
        return SetReadout(readout_config.config)
    else:
        raise ValueError("Unsupported readout function! ({})".format(readout_config.function))

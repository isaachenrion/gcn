from torch.autograd import Variable
import torch
from torch.utils.data import Dataset
import copy
import networkx as nx
import numpy as np
from collections import namedtuple
from .graph_utils import *
from datasets import GraphDataset

class FillGapsDataset(FixedOrderGraphDataset):
    def __init__(
            self,
            graphs=None,
            problem_type=None,
            vertex_dim=None,
            edge_dim=None,
            graph_targets=None,
            order=None,
        ):

        new_graphs = self.rebuild_graphs(graphs)

        super().__init__(
            graphs=new_graphs,
            problem_type=problem_type,
            vertex_dim=vertex_dim,
            edge_dim=edge_dim,
            graph_targets=graph_targets,
            order=order,
        )

    def rebuild_graphs(self, graphs):
        new_graphs = []
        for G_ in new_graphs:
            G = copy.deepcopy(G_)
            for u in G.nodes():
                x = G.node[u]['data']
                x_ = np.concatenate((x, 1), -1)
                x_null = np.zeros_like(x_)
                G.node[u]['data'] = x_
                G.node[u]['null'] = x_null
            new_graphs.append(G)
        return new_graphs

    def generate_missing_vertices(self, n_missing=1):
        missing_vertices = np.random.choice(self.order, n_missing)
        one_hot = np.zeros(self.order)
        one_hot[missing_vertices] = 1
        return one_hot
    

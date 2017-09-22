from torch.autograd import Variable
import torch
from torch.utils.data import Dataset
import copy
import networkx as nx
import numpy as np
from collections import namedtuple
from .graph_utils import *


class GraphDataset(Dataset):
    def __init__(self, vertices, edges, targets, problem_type, target_names):
        super().__init__()
        self.vertices_np = vertices
        self.targets_np = targets
        self.edges_np = edges
        self.vertices = None
        self.targets = None
        self.length, self.order, self.vertex_dim = vertices.shape

        self.problem_type = problem_type
        self.target_names = target_names
        self.batch_size = None
        self.ndim = len(self.vertices_np.shape)

        if edges is not None:
            self.dads_np = build_dads(edges)
        else:
            self.dads_np = build_complete_dads(self.length, self.order)
        self.dads = None

    def cuda(self):
        try:
            assert self.vertices is not None
            self.vertices = self.vertices.cuda()
            self.targets = self.targets.cuda()
            self.dads = self.dads.cuda()
        except AssertionError:
            raise Error("Data was not initialized. Cannot move to CUDA")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.vertices[index], self.targets[index], self.dads[index])

    def initialize(self, batch_size):
        n_batches, remainder = np.divmod(self.length, batch_size)
        n_batches = int(n_batches)
        remainder = int(remainder)
        self.n_batches = n_batches
        self.batch_size = batch_size
        if remainder != 0:
            self.vertices = torch.from_numpy(self.vertices_np)[:-remainder].contiguous().view(n_batches, batch_size, self.order, self.vertex_dim)
            self.targets = torch.from_numpy(self.targets_np)[:-remainder].contiguous()
            self.dads = torch.from_numpy(self.dads_np)[:-remainder].contiguous().view(n_batches, batch_size, self.order, self.order)
        else:
            self.vertices = torch.from_numpy(self.vertices_np).contiguous().view(n_batches, batch_size, self.order, self.vertex_dim)
            self.targets = torch.from_numpy(self.targets_np).contiguous()
            self.dads = torch.from_numpy(self.dads_np).contiguous().view(n_batches, batch_size, self.order, self.order)

        self.vertices = Variable(self.vertices).float()
        self.targets = Variable(self.targets).float()
        self.dads = Variable(self.dads).float()


class RegressionDataset(GraphDataset):
    def __init__(self, vertices, edges, targets, target_names):
        super().__init__(vertices, edges, targets, 'reg', target_names)
        assert len(targets.shape) == 2
        self.target_dim = targets.shape[1]

    def initialize(self, batch_size):
        super().initialize(batch_size)
        self.targets = self.targets.view(self.n_batches, batch_size, self.target_dim)

class ClassificationDataset(GraphDataset):
    def __init__(self, vertices, edges, targets, target_names):
        super().__init__(vertices, edges, targets, 'clf', target_names)
        assert len(targets.shape) == 1

    def initialize(self, batch_size):
        super().initialize(batch_size)
        self.targets = self.targets.view(n_batches, batch_size)
        self.targets = self.targets.long()

class VertexPredictionDataset(GraphDataset):
    def __init__(self, vertices, edges, targets, target_names):
        super().__init__(vertices, edges, targets, 'reg', target_names)
        assert len(targets.shape) == 3
        _, self.order, self.target_dim = targets.shape

    def initialize(self, batch_size):
        super().initialize(batch_size)
        self.targets = self.targets.view(self.n_batches, batch_size, self.order, self.target_dim)

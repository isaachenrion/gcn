from torch.autograd import Variable
import torch
from torch.utils.data import Dataset
import copy
import networkx as nx
import numpy as np
from collections import namedtuple
from .graph_utils import *


class GraphDataset(Dataset):
    def __init__(
            self,
            graphs=None,
            problem_type=None,
            vertex_dim=None,
            edge_dim=None,
            target_names=None,
            order=None,
        ):
        super().__init__()
        self.graphs = graphs
        self.original_graphs = graphs
        self.order = order
        self.problem_type = problem_type
        self.vertex_dim = vertex_dim
        self.edge_dim = edge_dim
        self.target_names = target_names
        self.is_batched = False


    def set_graphs(self, graphs):
        assert len(graphs) == len(self)
        self.graphs = graphs

    def has_vertex_data(self):
        return self.vertex_dim > 0

    def has_edge_data(self):
        return self.edge_dim > 0

    def has_graph_data(self):
        return self.target_names is not None

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

    def preprocess(self):
        wrapped_dataset = copy.deepcopy(self)
        for i, G in enumerate(wrapped_dataset):
            if self.has_vertex_data():
                wrap_vertices(G)
            if self.has_edge_data():
                wrap_edges(G)
            if self.has_graph_data():
                dtype = 'float' if self.problem_type == 'reg' else 'long'
                wrap_target_names(G, self.target_names, dtype)
            return wrapped_dataset

    def pad_graphs(self):
        '''Pad all the graphs with zero edges - make them fully connected'''
        for i, G in enumerate(self.original_graphs):
            G_ = fully_connected_padding(G)
            self.graphs[i] = G_


class FixedOrderGraphDataset(GraphDataset):
    def __init__(self, flat_graph_state_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.flat_graph_state_dim = flat_graph_state_dim

    def preprocess(self):
        wrapped_dataset = super().preprocess()
        for i, G in enumerate(wrapped_dataset):
            wrap_one_graph_state(G, 'flat_graph_state')

class BatchedGraphDataset(GraphDataset):
    def __init__(self, graph_dataset, batch_size):
        super().__init__(
            graphs=graph_dataset.graphs,
            problem_type=graph_dataset.problem_type,
            vertex_dim=graph_dataset.vertex_dim,
            edge_dim=graph_dataset.edge_dim,
            target_names=graph_dataset.target_names,
            order=graph_dataset.order,
        )
        self.make_batches(batch_size)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]

    def make_batches(self, batch_size):
        self.batch_size = batch_size
        self.n_batches, self.remainder = divmod(len(self.graphs), batch_size)
        self.batches = [[] for _ in range(self.n_batches)]
        on_batch = 0

        for i, G in enumerate(self.graphs):
            self.batches[on_batch].append(G)
            if (i + 1) % batch_size == 0:
                on_batch += 1
            if batch_size * on_batch + self.remainder == len(self.graphs):
                break

    def preprocess(self):
        wrapped_dataset = copy.deepcopy(self)
        for i, batch in enumerate(wrapped_dataset):
            for G in batch:
                if self.has_vertex_data():
                    wrap_vertices(G)
                if self.has_edge_data():
                    wrap_edges(G)
                if self.has_graph_data():
                    dtype = 'float' if self.problem_type == 'reg' else 'long'
                    wrap_target_names(G, self.target_names, dtype)
        return wrapped_dataset

    def randomize_nodes(self):

        old_batches = self.batches
        new_batches = []
        for batch in old_batches:
            old_batch = batch
            new_batch = nx.create_empty_copy(batch)
            permutation = np.random.permutation(self.order)
            # permute the vertex data
            if self.has_vertex_data():
                for i in old_batch.nodes():
                    new_batch.node[i]['data'] = old_batch.node[permutation[i]]['data']

            # permute the edge data
            if self.has_edge_data():
                for (u, v) in old_batch.edges():
                    u_, v_ = permutation[u], permutation[v]
                    new_batch.add_edge(u, v, data=old_batch.edge[u_][v_]['data'])

            # port over the graph level info
            for k, v in old_batch.graph.items():
                new_batch.graph[k] = v

            new_batches.append(new_batch)
        self.batches = new_batches


class BatchedFixedOrderGraphDataset(BatchedGraphDataset):
    def __init__(self, graph_dataset, batch_size):
        self.flat_graph_state_dim=graph_dataset.flat_graph_state_dim
        self.model_graph = nx.create_empty_copy(graph_dataset.graphs[0])

        super().__init__(graph_dataset, batch_size)

    def make_batches(self, batch_size):

        self.batch_size = batch_size
        self.n_batches, self.remainder = divmod(len(self.graphs), batch_size)

        def create_empty_batch_graph(batch_size):
            empty_batch_graph = nx.create_empty_copy(self.model_graph)
            if self.vertex_dim > 0:
                add_vertex_data(empty_batch_graph, np.zeros([self.order, batch_size, self.vertex_dim]))
            if self.edge_dim > 0:
                add_edge_data(empty_batch_graph, np.zeros([self.order, self.order, batch_size, self.edge_dim]))

            if self.target_names is not None:
                if self.problem_type == 'clf':
                    graph_data_dict = {target.name: np.zeros([batch_size]) for target in self.target_names}
                else:
                    graph_data_dict = {target.name: np.zeros([batch_size, target.dim]) for target in self.target_names}


                add_graph_data_dict(empty_batch_graph, graph_data_dict)

            add_graph_data(empty_batch_graph, np.zeros([batch_size, self.flat_graph_state_dim]), 'flat_graph_state')

            empty_batch_graph.graph['batch_size'] = batch_size

            return empty_batch_graph

        self.batches = [create_empty_batch_graph(self.batch_size) for _ in range(self.n_batches)]

        on_batch = 0

        for i, G in enumerate(self.graphs):
            batch_idx = i % self.batch_size
            if self.vertex_dim > 0:
                for u in range(self.order):
                    self.batches[on_batch].node[u]['data'][batch_idx] = G.node[u]['data']

            if self.edge_dim > 0:
                for u, v in G.edges():
                    self.batches[on_batch].edge[u][v]['data'][batch_idx] = G.edge[u][v]['data']

            if self.target_names is not None:
                for target in self.target_names:
                    self.batches[on_batch].graph[target.name][batch_idx] = G.graph[target.name]


            self.batches[on_batch].graph['flat_graph_state'][batch_idx] = G.graph['flat_graph_state']
            if batch_idx + 1 == batch_size:
                on_batch += 1
            if batch_size * on_batch + self.remainder == len(self.graphs):
                break

    def preprocess(self):
        wrapped_dataset = copy.deepcopy(self)
        for i, G in enumerate(wrapped_dataset):
            if self.has_vertex_data():
                wrap_vertices(G)
            if self.has_edge_data():
                wrap_edges(G)
            if self.has_graph_data():
                dtype = 'float' if self.problem_type == 'reg' else 'long'
                wrap_target_names(G, self.target_names, dtype)
            wrap_one_graph_state(G, 'flat_graph_state')

        return wrapped_dataset

    def randomize_nodes(self):

        old_batches = self.batches
        new_batches = []
        for batch in old_batches:
            old_batch = batch
            new_batch = nx.create_empty_copy(batch)
            permutation = np.random.permutation(self.order)
            # permute the vertex data
            if self.has_vertex_data():
                for i in old_batch.nodes():
                    new_batch.node[i]['data'] = old_batch.node[permutation[i]]['data']

            # permute the edge data
            if self.has_edge_data():
                for (u, v) in old_batch.edges():
                    u_, v_ = permutation[u], permutation[v]
                    new_batch.add_edge(u, v, data=old_batch.edge[u_][v_]['data'])

            # port over the graph level info
            for k, v in old_batch.graph.items():
                new_batch.graph[k] = v

            new_batches.append(new_batch)
        self.batches = new_batches


class GraphDataset2(Dataset):
    def __init__(self, graphs, targets, problem_type, target_names):
        super().__init__()
        self.graphs_np = graphs
        self.targets_np = targets
        self.graphs = None
        self.targets = None
        self.length, self.order, self.vertex_dim = graphs.shape
        _, self.target_dim = targets.shape

        self.problem_type = problem_type
        self.target_names = target_names
        self.batch_size = None

        self.dads_np = self.build_dads(self.length, self.order)
        self.dads = None

    @staticmethod
    def build_dads(n, order):
        A_tilde = np.ones((order, order))
        D_tilde = np.sum(A_tilde, 1, keepdims=True)
        D_tilde_sqrt = np.power(D_tilde, -0.5)
        dad = D_tilde_sqrt * A_tilde * D_tilde_sqrt
        return np.tile(dad, (n, 1, 1))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.graphs[index], self.targets[index], self.dads[index])

    def initialize(self, batch_size):
        n_batches, remainder = np.divmod(self.length, batch_size)
        n_batches = int(n_batches)
        remainder = int(remainder)
        self.batch_size = batch_size
        self.graphs = torch.from_numpy(self.graphs_np)[:-remainder].contiguous().view(n_batches, batch_size, self.order, self.vertex_dim)
        self.targets = torch.from_numpy(self.targets_np)[:-remainder].contiguous().view(n_batches, batch_size, self.target_dim)
        self.dads = torch.from_numpy(self.dads_np)[:-remainder].contiguous().view(n_batches, batch_size, self.order, self.order)

        self.graphs = Variable(self.graphs).float()
        self.targets = Variable(self.targets).float()
        self.dads = Variable(self.dads).float()
        #import ipdb; ipdb.set_trace()

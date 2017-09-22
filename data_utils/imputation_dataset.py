import torch
from torch.autograd import Variable
from .datasets import VertexPredictionDataset
import numpy as np
from .graph_utils import Target

class ImputationDataset(VertexPredictionDataset):
    def __init__(self, dataset, missing_prob):
        vertices = dataset.vertices_np
        edges = dataset.edges_np
        self.missing_prob = missing_prob
        target_names=[Target(name=str(i), dim=dataset.vertex_dim, type='vertex') for i in range(dataset.order)]
        vertices_padded = np.concatenate((vertices, np.ones((vertices.shape[0], vertices.shape[1], 1))), -1)

        super().__init__(
                vertices=vertices_padded,
                edges=edges,
                targets=vertices,
                target_names=target_names
                )

    def missing_values_mask(self):
        #mask = Variable(torch.ones((self.batch_size, self.order, 1)))
        #mask[:, 0, 0] = 0
        #mask = Variable(torch.zeros((self.batch_size, self.order, 1)))
        mask = Variable(torch.bernoulli((1-self.missing_prob) * torch.ones(self.batch_size, self.order, 1)))
        if torch.cuda.is_available(): mask = mask.cuda()
        return mask

    def __getitem__(self, index):
        mask = self.missing_values_mask()
        out = (mask * self.vertices[index], self.targets[index], self.dads[index])
        return out

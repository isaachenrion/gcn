from .global_imports import *

def qm7_ng(prefix, representation='edge', N=-1):
    n_atoms = 23
    target_names = [
        Target('E', 'graph', 1)
    ]

    data = scipy.io.loadmat(os.path.join(DATA_DIR, "qm7.mat"))
    targets = np.expand_dims(data['T'][0], -1)

    # standardize
    edge_data = np.expand_dims(data['X'], -1)
    edge_data = edge_data - np.mean(edge_data, 0, keepdims=True)
    edge_data = edge_data / np.std(edge_data, 0, keepdims=True)

    vertex_data = np.concatenate([np.expand_dims(data['Z'],-1), data['R']], -1)
    vertex_data = vertex_data - np.mean(vertex_data, 0, keepdims=True)
    vertex_data = vertex_data / np.std(vertex_data, 0, keepdims=True)

    P = data['P']

    if prefix == 'train':
        good_indices = np.concatenate(P[1:], -1)[:N]
    elif prefix == 'eval':
        good_indices = P[0][:N]

    graphs = vertex_data

    processed_data = GraphDataset2(
        graphs=graphs,
        targets=targets,
        problem_type="reg",
        target_names=target_names
    )
    return processed_data

def qm7_small(prefix, representation='edge'):
    return qm7(prefix, representation,100)

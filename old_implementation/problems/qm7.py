from .global_imports import *

def qm7(prefix, representation='edge', N=-1):
    n_atoms = 23
    if representation == 'vertex':
        vertex_dim = 4
        edge_dim = 0
    elif representation == 'edge':
        vertex_dim = 0
        edge_dim = 1
    target_names = [
        Target('E', 'graph', 1)
    ]

    data = scipy.io.loadmat(os.path.join(DATA_DIR, "qm7.mat"))
    T = np.expand_dims(data['T'][0], -1)

    # standardize
    edge_data = np.expand_dims(data['X'], -1)
    edge_data = edge_data - np.mean(edge_data, 0, keepdims=True)
    edge_data = edge_data / np.std(edge_data, 0, keepdims=True)

    vertex_data = np.concatenate([np.expand_dims(data['Z'],-1), data['R']], -1)
    vertex_data = vertex_data - np.mean(vertex_data, 0, keepdims=True)
    vertex_data = vertex_data / np.std(vertex_data, 0, keepdims=True)

    P = data['P']

    graphs = []
    if prefix == 'train':
        good_indices = np.concatenate(P[1:], -1)[:N]
    elif prefix == 'eval':
        good_indices = P[0][:N]

    for i, idx in enumerate(good_indices):
        G = nx.complete_graph(n_atoms)
        if representation == 'vertex':
            add_vertex_data(G, vertex_data[idx])
            add_graph_data(G, np.reshape(vertex_data[idx], [-1]), key='flat_graph_state')

        elif representation == 'edge':
            add_edge_data(G, edge_data[idx])
            add_graph_data(G, np.reshape(edge_data[idx], [-1]), key='flat_graph_state')

        add_graph_data(G, T[idx], key='E')
        add_graph_data(G, vertex_dim, key='edge_dim')
        add_graph_data(G, edge_dim, key='vertex_dim')
        graphs.append(G)

    processed_data = FixedOrderGraphDataset(
        order=23,
        graphs=graphs,
        flat_graph_state_dim=G.graph['flat_graph_state'].shape[-1],
        problem_type="reg",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        target_names=target_names
    )
    return processed_data

def qm7_small(prefix, representation='edge'):
    return qm7(prefix, representation,100)

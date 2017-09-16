from .global_imports import *
def qm7b(prefix, N=-1):
    n_atoms = 23

    data = scipy.io.loadmat(os.path.join(DATA_DIR, "qm7b.mat"), chars_as_strings=True)
    edge_data = np.expand_dims(data['X'], -1)

    # standardize
    edge_data = edge_data - np.mean(edge_data, 0, keepdims=True)
    edge_data = edge_data / np.std(edge_data, 0, keepdims=True)

    targets = data['T']
    vertex_data = np.ones((len(edge_data), 1))

    target_names = [
        Target('E-PBE0', 'graph', 1),
        Target('E-max-ZINDO', 'graph', 1),
        Target('I-max-ZINDO', 'graph', 1),
        Target('HOMO-ZINDO', 'graph', 1),
        Target('LUMO-ZINDO', 'graph', 1),
        Target('E-1st-ZINDO', 'graph', 1),
        Target('IP-ZINDO', 'graph', 1),
        Target('EA-ZINDO', 'graph', 1),
        Target('HOMO-PBE0', 'graph', 1),
        Target('LUMO-PBE0', 'graph', 1),
        Target('HOMO-GW', 'graph', 1),
        Target('LUMO-GW', 'graph', 1),
        Target('alpha-PBE0', 'graph', 1),
        Target('alpha-SCS', 'graph', 1),
    ]


    # make validation set
    q7data = scipy.io.loadmat(os.path.join(DATA_DIR, "qm7.mat"))
    P = q7data['P']
    if prefix == 'train':
        good_indices = np.concatenate(P[1:], -1)[:N]
    elif prefix == 'eval':
        good_indices = P[0][:N]

    graphs = []
    for i, idx in enumerate(good_indices):
        G = nx.Graph()
        add_edge_data(G, edge_data[idx])
        #add_graph_data(G, T[idx], key='readout')
        add_graph_data(G, np.reshape(edge_data[idx], [-1]), key='flat_graph_state')
        graph_data_dict = {target.name: T[idx][j] for j, target in enumerate(target_names)}
        add_graph_data_dict(G, graph_data_dict)

        graphs.append(G)



    processed_data = FixedOrderGraphDataset(
        order=n_atoms,
        graphs=graphs,
        flat_graph_state_dim=G.graph['flat_graph_state'].shape[-1],
        problem_type="reg",
        vertex_dim=0,
        edge_dim= 1,
        target_names=target_names,

    )
    return processed_data

def qm7b_small(prefix):
    return qm7b(prefix, 100)

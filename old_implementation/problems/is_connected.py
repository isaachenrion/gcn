from .global_imports import *
def is_connected(prefix='train', num_examples=1, args=None):

    target_names = [
        Target('is_connected', 'graph', 2)
    ]
    graphs = []
    vertex_dim = 1
    edge_dim = 1
    mean_connected = 0.
    for i in range(num_examples):
        min_order = 1
        max_order = args.max_order if args is not None else 10
        if prefix == 'eval':
            min_order = max_order; max_order = 2 * max_order
        order = np.random.randint(min_order, max_order + 1)
        eps = 1e-1 * (4 * np.random.randint(0, 2))
        p = (1 + eps) * np.log(order) / order
        G= nx.fast_gnp_random_graph(order, p)
        G.graph['is_connected'] = np.ones([1]) if nx.is_connected(G) else np.zeros([1])

        fgs = np.zeros([max_order, max_order])
        for u in G.nodes():
            G.node[u]['data'] = np.array([[1]])
        for u, v in G.edges():
            G.edge[u][v]['data'] = np.array([[1]])
            fgs[u][v] = 1

        # padding
        for u in range(order, max_order):
            G.add_node(u, data=np.array([[0]]))

        G.graph['flat_graph_state'] = np.reshape(fgs, [-1])
        mean_connected += G.graph['is_connected']
        graphs.append(G)
    mean_connected /= num_examples
    logging.info(mean_connected)
    data = FixedOrderGraphDataset(
        order=max_order,
        graphs=graphs,
        problem_type="clf",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        flat_graph_state_dim=G.graph['flat_graph_state'].shape[-1],
        target_names=target_names,
    )
    return data

def is_connected_padded(prefix='train', num_examples=1):
    data = is_connected(prefix, num_examples)
    data.pad_graphs()

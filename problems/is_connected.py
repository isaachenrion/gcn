from .global_imports import *
def is_connected(prefix='train', num_examples=1, args=None):

    target_names = [
        Target('is_connected', 'graph', 2)
    ]
    #data = []
    targets = []
    edges = []
    vertices = []
    #vertex_dim = 1

    #edge_dim = 1
    #mean_connected = 0.
    for i in range(num_examples):
        min_order = 1
        max_order = args.max_order if args is not None else 10
        if prefix == 'eval':
            min_order = max_order; max_order = 2 * max_order

        #order = np.random.randint(min_order, max_order + 1)
        order = args.max_order

        eps = 1e-1 * (4 * np.random.randint(0, 2))
        p = (1 + eps) * np.log(order) / order
        G= nx.fast_gnp_random_graph(order, p)

        targets.append(np.ones([1]) if nx.is_connected(G) else np.zeros([1]))
        edges.append(nx.to_numpy_matrix(G))
        vertices.append(np.ones(order, 1))

    targets = np.array(targets)
    edges = np.array(edges)
    vertices = np.array(vertices)

    #mean_connected /= num_examples
    #logging.info(mean_connected)
    data = GraphDataset2(
        vertices=vertices,
        edges=edges,
        targets=targets,
        problem_type="clf",
        target_names=target_names,
    )
    return data

def is_connected_padded(prefix='train', num_examples=1):
    data = is_connected(prefix, num_examples)
    data.pad_graphs()

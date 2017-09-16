from .global_imports import *

def has_path(prefix='train', num_examples=1, debug=False):

    target_names = [
        Target('has_path', 'graph', 1)
    ]
    graphs = []
    vertex_dim = 1
    edge_dim = 0
    readout_dim = 1
    for i in range(num_examples):
        order = np.random.randint(5, 15)
        graph = nx.fast_gnp_random_graph(order, 0.5)
        source = np.random.randint(0, order)
        target = np.random.randint(0, order)
        for node in graph.nodes():
            if node in [source, target]:
                graph.node[node]['data'] = np.ones([1, 1])
            else:
                graph.node[node]['data'] = np.zeros([1, 1])
        graph.graph['has_path'] = np.ones([1]) if nx.has_path(graph, source, target) else np.zeros([1])

        graphs.append(graph)
    data = GraphDataset(
        graphs=graphs,
        problem_type="clf",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        target_names=target_names,
    )
    return data

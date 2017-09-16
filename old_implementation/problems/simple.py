from .global_imports import *

def simple(prefix='train', num_examples=1):
    target_names = [
        Target('result', 'graph', 2)
    ]
    graphs = []
    vertex_dim = 1
    edge_dim = 0
    mean_connected = 0.
    for i in range(num_examples):
        order = np.random.randint(5, 15)
        eps = 1e-5 * (2 * np.random.randint(0, 2) - 1)
        p = (1 + eps) * np.log(order) / order
        graph= nx.fast_gnp_random_graph(order, p)

        for node in graph.nodes():
            graph.node[node]['data'] = np.random.randint(0, 2, [1,1])
        graph.graph['result'] = np.ones([1]) if graph.node[0]['data'][0] == 0 else np.zeros([1])

        print(graph.graph['result'])
        graphs.append(graph)
    data = GraphDataset(
        graphs=graphs,
        problem_type="clf",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        target_names=target_names,
    )
    return data

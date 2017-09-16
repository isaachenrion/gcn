
from .global_imports import *

def arithmetic(prefix='train', num_examples=1, debug=False):

    target_names = [
        Target('result', 'graph', 2)
    ]
    graphs = []
    vertex_dim = 5
    edge_dim = 3
    for i in range(num_examples):
        order = np.random.randint(5, 15)
        p = 0.5
        graph= nx.fast_gnp_random_graph(order, p)

        readout = 0.0
        for node in graph.nodes():
            node_data = np.reshape(np.random.randint(0, 2, vertex_dim), [1, -1])
            graph.add_node(node)
            graph.node[node]['data'] = node_data
            graph.add_edge(node, node, data=np.zeros([1, edge_dim]))

        for u, v in graph.edges():
            edge_matrix = np.expand_dims(2 * np.random.randint(0, 2, [vertex_dim, edge_dim]) - 1, 0)
            edge_data = np.matmul(graph.node[u]['data'] * graph.node[v]['data'], edge_matrix)
            graph.edge[u][v]['data'] = np.reshape(edge_data, [1, -1])
            graph.edge[u][v]['matrix'] = edge_matrix
            readout += edge_data.sum()

        graph.graph['result'] = np.expand_dims((readout > 0).astype('float32'), 1)

        graphs.append(graph)

    data = GraphDataset(
        graphs=graphs,
        problem_type="clf",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        target_names=target_names
    )
    return data

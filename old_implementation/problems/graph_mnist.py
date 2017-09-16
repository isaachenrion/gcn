from .global_imports import *


class MNISTinFour(FixedOrderGraphDataset):
    def __init__(self, mode):
        data_path = os.path.join(DATA_DIR, 'mnist')
        if mode == 'train':
            X = np.load(os.path.join(data_path, 'x-train.npy'))
            Y = np.load(os.path.join(data_path, 'y-train.npy'))
        elif mode == 'eval':
            X = np.load(os.path.join(data_path, 'x-test.npy'))
            Y = np.load(os.path.join(data_path, 'y-test.npy'))
        else:
            raise ValueError("mode must be train or eval!")

        graphs = []
        image_size = 28
        half = int(image_size / 2)
        for x, y in zip(X, Y):
            G = nx.Graph()
            x_tl = np.reshape(x[:half, :half], [-1])
            x_tr = np.reshape(x[:half, half:image_size], [-1])
            x_bl = np.reshape(x[half:image_size, :half], [-1])
            x_br = np.reshape(x[half:image_size, half:image_size], [-1])

            G.add_node(0, data=x_tl)
            G.add_node(1, data=x_tr)
            G.add_node(2, data=x_bl)
            G.add_node(3, data=x_br)

            G.add_edge(0, 1)
            G.add_edge(0, 2)
            G.add_edge(1, 3)
            G.add_edge(2, 3)

            G.graph['digit'] = y
            G.graph['flat_graph_state'] = np.reshape(x, [-1])

            graphs.append(G)


        super().__init__(
            graphs=graphs,
            problem_type='clf',
            vertex_dim=int((image_size / 2) ** 2),
            edge_dim=0,
            target_names=[Target('digit', 'graph', 10)],
            order=4,
            flat_graph_state_dim=image_size * image_size
        )

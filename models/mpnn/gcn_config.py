from .mpnn_config import *

GCNConfig = namedtuple(
        'GCNConfig', [
            'message',
            'vertex_update',
            'readout',
            'embedding',
            'n_iters',
            'parallelism',
            'mp_prob',
        ]
)

def get_gcn_config(args, dataset):
    #if args.message == 'constant':
    #    args.message_dim = args.hidden_dim + dataset.edge_dim
    config = GCNConfig(
        message=FunctionAndConfig(
            function=args.message,
            config=MessageConfig(
                hidden_dim=args.hidden_dim,
                edge_dim=0,
                message_dim=args.hidden_dim,
            )
        ),
        vertex_update=FunctionAndConfig(
            function=args.vertex_update,
            config=VertexUpdateConfig(
                vertex_state_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim,
                message_dim=args.hidden_dim
            )
        ),
        readout=FunctionAndConfig(
            function=args.readout,
            config=ReadoutConfig(
                hidden_dim=args.hidden_dim,
                readout_hidden_dim=10,
                mode=dataset.problem_type,
                target_names=dataset.target_names,
            )
        ),
        embedding=FunctionAndConfig(
            function=args.embedding,
            config=EmbeddingConfig(
                data_dim=dataset.vertex_dim,
                state_dim=args.hidden_dim
            )
        ),
        n_iters=args.n_iters,
        parallelism=args.parallelism,
        mp_prob=args.mp_prob
    )
    return config

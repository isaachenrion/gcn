from collections import namedtuple

FlatConfig = namedtuple(
        'FlatConfig', [
            'state_dim',
            'hidden_dim',
            'target_names',
            'mode'
        ]
)

def get_flat_config(args, dataset):
    config = FlatConfig(
        state_dim=(dataset.vertex_dim * dataset.order),
        hidden_dim=args.hidden_dim,
        target_names=dataset.target_names,
        mode=dataset.problem_type
    )
    return config

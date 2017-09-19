from collections import namedtuple
from .mpnn import make_mpnn, get_mpnn_config,  make_gcn, get_gcn_config
from .flat import make_flat, get_flat_config

def get_generator_and_config(model_str, args, dataset):
    if model_str in ['mpnn', 'mpnn_set', 'vcn', 'dtnn']:
        return make_mpnn, get_mpnn_config(args, dataset)
    elif model_str == 'flat':
        return make_flat, get_flat_config(args, dataset)
    elif model_str in ['gcn1', 'gcn2']:
        return make_gcn, get_gcn_config(args, dataset)

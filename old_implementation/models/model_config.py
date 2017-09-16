from collections import namedtuple
from .mpnn import make_mpnn, get_mpnn_config, GCN, get_gcn_config
from .flat import make_flat, get_flat_config

def get_model_generator(model_str):
    if model_str in ['mpnn', 'mpnn_set', 'vcn', 'dtnn']:
        return make_mpnn
    elif model_str == 'flat':
        return make_flat
    elif model_str == 'gcn':
        return GCN

def get_model_config(model_str, args, dataset):
    if model_str in ['mpnn', 'mpnn_set', 'vcn', 'dtnn']:
        return get_mpnn_config(args, dataset)
    elif model_str == 'flat':
        return get_flat_config(args, dataset)
    elif model_str == 'gcn':
        return get_gcn_config(args, dataset)

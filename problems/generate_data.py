import pickle
import os
from config import DATA_DIR

from .arithmetic import *
from .has_path import *
from .is_connected import *
from .simple import *
from .qm7 import *
from .qm7b import *
from .qm7_ng import *
from .graph_mnist import MNISTinFour

def generate_data(prefix, args):
    if prefix == 'train':
        num_examples = args.n_train
    else: num_examples = args.n_eval
    if args.problem == 'arithmetic':
        data = arithmetic(prefix, num_examples)
    elif args.problem == 'has_path':
        data = has_path(prefix, num_examples)
    elif args.problem == 'is_connected':
        data = is_connected(prefix, num_examples, args)
    elif args.problem == 'qm7_edge_representation':
        data = qm7(prefix, 'edge')
    elif args.problem == 'qm7_ng':
        data = qm7_ng(prefix, 'edge')
    elif args.problem == 'qm7_edge_representation_small':
        data = qm7_small(prefix, 'edge')
    elif args.problem == 'qm7_vertex_representation':
        data = qm7(prefix, 'vertex')
    elif args.problem == 'qm7_vertex_representation_small':
        data = qm7_small(prefix, 'vertex')
    elif args.problem == 'qm7b':
        data = qm7b(prefix)
    elif args.problem == 'qm7b_small':
        data = qm7b_small(prefix)
    elif args.problem == 'simple':
        data = simple(prefix, num_examples)
    elif args.problem == 'mnist':
        data = MNISTinFour(prefix)
    else:
        raise ValueError("Problem was not recognised.")

    with open(os.path.join(DATA_DIR,'{}-{}.pkl'.format(args.problem, prefix)), 'wb') as f:
        pickle.dump(data, f)
    return None

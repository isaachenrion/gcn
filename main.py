import torch
from problems import generate_data
from train import train
import argparse
import os
import logging


parser = argparse.ArgumentParser(description='MPNN')

# data args
parser.add_argument('--n_train', type=int, default=10000, help='Number of training examples to generate.')
parser.add_argument('--n_eval', type=int, default=2000, help='Number of test examples to generate.')
parser.add_argument('--problem', '-t', type=int, default=0, help='task to train on')
parser.add_argument('--gen', action='store_true', help='generate the data')
parser.add_argument('--max_order', type=int, default=10, help='order of graphs to generate')
parser.add_argument('--missing_prob', type=float, default=0., help='vertex dropout probability (for imputation)')
parser.add_argument('--imp', action='store_true', help='is this an imputation problem?')
# miscellaneous args
parser.add_argument('--gpu', type=int, default=0, help='Device to use (GPU)')
parser.add_argument('--verbosity', '-v', type=int, default=1)
parser.add_argument('--debug', action='store_true')

# training args
parser.add_argument('--epochs', '-e', type=int, default=500, help='number of epochs to train')
parser.add_argument('--batch_size', '-b', type=int, default=100, help='batch size for training')
parser.add_argument('--randomize_nodes', '-r', action='store_true', help='Randomize the ordering of the nodes')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00005, help='L2 weight decay')

# model args
parser.add_argument('--load', '-l', default=None)
parser.add_argument('--model', '-m', type=int, default=0)

# model dimension args
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--message_dim', type=int, default=100)
parser.add_argument('--vertex_state_dim', type=int, default=0)

# mpnn args
parser.add_argument('--n_iters', type=int, default=1, help='Number of iterations of message passing')
parser.add_argument('--parallelism', '-p', type=int, default=1, help='MPNN parallelism level (0, 1)')
parser.add_argument('--mp_prob', type=float, default=1, help='Probability to do message passing at a vertex')
parser.add_argument('--readout', default=None)
parser.add_argument('--message', default=None)
parser.add_argument('--vertex_update', default=None)
parser.add_argument('--embedding', default=None)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if args.debug:
    if args.problem == 0: args.problem = 1
    args.weight_decay = 0.
    args.hidden_dim = 7
    #args.n_iters = 2
    args.message_dim = 11
    args.batch_size = 23

PROBLEMS = [
    'qm7_edge_representation',
    'qm7_edge_representation_small',
    'qm7_vertex_representation',
    'qm7_vertex_representation_small',
    'qm7b',
    'qm7b_small',
    'arithmetic',
    'has_path',
    'is_connected',
    'mnist',
    'simple',

]
args.problem = PROBLEMS[args.problem]

MODELS = [
'flat', 'gcn1', 'gcn2'
]
args.model = MODELS[args.model]
if args.model == 'gcn1':
    args.message = 'dtnn' # 'fully_connected'
    args.vertex_update = 'gru'
    args.embedding = 'fully_connected'
elif args.model == 'gcn2':
    args.message = 'dtnn' # 'fully_connected'
    args.vertex_update = 'gru'
    args.embedding = 'fully_connected'
elif args.model == 'flat':
    args.readout = None
    args.message = None
    args.vertex_update = None
    args.embedding = None
if args.imp:
    args.readout = 'vertex'
else:
    args.readout = 'dtnn'

def main():


    if args.gen:
        generate_data('train', args)
        generate_data('eval', args)

    model = train(args)

if __name__ == "__main__":
    main()

import os
import pickle
from .imputation_dataset import ImputationDataset
from .add_virtual_node import add_virtual_node, add_target_nodes
from config import DATA_DIR

def load_from_path(data_path, args):
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    if args.model == 'vcn':
        dataset.add_target_nodes()
    if args.imp:
        dataset = ImputationDataset(dataset, args.missing_prob)

    dataset.initialize(args.batch_size)
    return dataset


def load_data(args):
    train_data_path = os.path.join(DATA_DIR, args.problem + '-train.pkl')
    eval_data_path = os.path.join(DATA_DIR, args.problem + '-eval.pkl')

    training_set = load_from_path(train_data_path, args)
    validation_set = load_from_path(eval_data_path, args)

    return training_set, validation_set

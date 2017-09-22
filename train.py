
"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from experiment_config import get_experiment_config
import numpy as np
import time
import datetime
import itertools
import gc
import copy
import os
from models import *
from data_utils import load_data
import networkx as nx
import torch.nn.functional as F
import sys
import logging
from capturing import Capturing
from collections import OrderedDict


EXP_DIR = "experiments"

def train(args):
    global DEBUG
    DEBUG = args.debug

    # get timestamp for model id
    dt = datetime.datetime.now()
    timestamp = '{}-{}/{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    model_dir = os.path.join(EXP_DIR, timestamp)
    os.makedirs(model_dir)

    # configure logging
    logging.basicConfig(filename=os.path.join(model_dir, 'log.txt'),level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    if args.verbosity >= 1:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # set device (if using CUDA)
    seed = 12345
    if torch.cuda.is_available():
        torch.cuda.device(args.gpu)
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    # write the args to outfile
    for k, v in sorted(vars(args).items()): logging.info('{} : {}\n'.format(k, v))

    # load data
    training_set, validation_set = load_data(args)

    logging.info('Loaded data: {} training examples, {} validation examples\n'.format(
        len(training_set), len(validation_set)))

    # get config
    experiment_config = get_experiment_config(args, training_set, validation_set)

    # initialize model
    if args.load is None:
        logging.info('Initializing model...\n')
        model = experiment_config.model_generator(experiment_config.model_config)
    else:
        logging.info('Loading model from {}\n'.format(args.load))
        model = torch.load(os.path.join(EXP_DIR, args.load, 'model.ckpt'))
    if torch.cuda.is_available():
        training_set.cuda()
        validation_set.cuda()
        model.cuda()
    logging.info(model)
    logging.info('Number of trainable parameters: {}\n'.format(model.number_of_parameters()))
    logging.info('Training loss: {}\n'.format(experiment_config.loss_fn))

    # optimizer
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5, patience=3, min_lr=lr/32)
    logging.info(optimizer)
    logging.info(scheduler)

    # Start Training
    for epoch in range(1, args.epochs + 1):
        if args.randomize_nodes:
            training_set.randomize_nodes()

        train_results = train_one_epoch(model, training_set, experiment_config.loss_fn, optimizer, experiment_config.monitors, args.debug)
        logging.info(results_str(epoch, train_results, 'train'))

        if epoch % 5 == 0:
            results = evaluate_one_epoch(model, validation_set, experiment_config.loss_fn, experiment_config.monitors)
            logging.info(results_str(epoch, results, 'eval'))

            torch.save(model, os.path.join(model_dir, 'model.ckpt'))
            logging.info("Saved model to {}\n".format(os.path.join(model_dir, 'model.ckpt')))

            logging.info("Training: processed {:.1f} graphs per second".format(len(training_set) / train_results['time']))

            with Capturing() as output:
                scheduler.step(results['loss'])
            if len(output) > 0:
                logging.info(output[0])

    return model

def results_str(epoch, results, run_mode):
    out_str = ""
    if run_mode == 'train':
        out_str += "Epoch {}\n".format(epoch)
    for k, v in results.items():
        out_str += "{} {}: {:.5f}\n".format(run_mode, k, v)
    if run_mode == 'eval':
        pad_str = '\n{s:{c}^{n}}\n'.format(s='#',n=20,c='#')
        out_str += pad_str
    return out_str

def unwrap(variable_dict):
    od = OrderedDict()
    for name, var in variable_dict.items():
        od[name] = var.data.cpu().numpy().item()
    return od

def train_one_batch(model, batch, loss_fn, optimizer, monitors):
    # zero optimizer
    optimizer.zero_grad()
    # get batch
    x, y, dads = batch
    # forward model
    model_output = model(x, dads)
    # get loss
    batch_loss = loss_fn(model_output, y)
    # backward and optimize
    batch_loss.backward(); optimizer.step()
    # get stats
    batch_stats = unwrap(monitors(model_output, y))
    return batch_stats

def train_one_epoch(model, dataset, loss_fn, optimizer, monitors, debug):
    t0 = time.time()
    epoch_stats = OrderedDict()
    for name in monitors.names:
        epoch_stats[name] = 0.0

    model.train()

    #import ipdb; ipdb.set_trace()
    for i, batch in enumerate(dataset):
        batch_stats = train_one_batch(model, batch, loss_fn, optimizer, monitors)
        for name in monitors.names:
            epoch_stats[name] = (epoch_stats[name] + batch_stats[name])

    for name, stat in epoch_stats.items():
        epoch_stats[name] = stat / dataset.n_batches

    epoch_stats["time"] = time.time() - t0

    gc.collect()
    return epoch_stats

def evaluate_one_batch(model, batch, monitors):
    # get batch
    x, y, dads = batch
    # forward model
    model_output = model(x, dads)
    # get stats
    batch_stats = unwrap(monitors(model_output, y))
    return batch_stats

def evaluate_one_epoch(model, dataset, loss_fn, monitors):
    t0 = time.time()
    epoch_stats = OrderedDict()
    for name in monitors.names:
        epoch_stats[name] = 0.

    model.eval()

    for i, batch in enumerate(dataset):
        batch_stats = evaluate_one_batch(model, batch, monitors)
        for name in monitors.names:
            epoch_stats[name] = epoch_stats[name] + batch_stats[name]

    for name, stat in epoch_stats.items():
        epoch_stats[name] = stat / dataset.n_batches
    epoch_stats["time"] = time.time() - t0

    return epoch_stats

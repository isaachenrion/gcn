
from collections import namedtuple
from monitors import classification_monitors, regression_monitors
from models import get_generator_and_config
import torch.nn as nn
from losses import CrossEntropy, MSEGraphLoss

ExperimentConfig = namedtuple(
        'ExperimentConfig', [
            'training_set',
            'validation_set',
            'model_generator',
            'model_config',
            'mode',
            'loss_fn',
            'monitors'
        ]
)

def get_experiment_config(args, training_set, validation_set):
    generator, config = get_generator_and_config(args.model, args, training_set)
    mode=training_set.problem_type

    if mode == 'clf': # classification
        loss_fn = CrossEntropy(
            target_names=training_set.target_names,
            )
        monitors = classification_monitors(args, validation_set)
    elif mode == 'reg': # regression
        loss_fn = MSEGraphLoss(
            target_names=training_set.target_names,
            )
        monitors = regression_monitors(args, validation_set)

    experiment_config = ExperimentConfig(
        model_generator=generator,
        model_config=config,
        training_set=training_set,
        validation_set=validation_set,
        mode=mode,
        loss_fn=loss_fn,
        monitors=monitors
    )
    return experiment_config

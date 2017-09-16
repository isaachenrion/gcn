import torch
import torch.nn.functional as F
from losses import *

def classification_monitors(args, dataset):
    monitors = LossCollection(
        primary_loss=CrossEntropy(
            target_names=dataset.target_names,
            as_dict=False
            ),
        other_losses=[
            Accuracy(
                target_names=dataset.target_names,
                as_dict=True
            )
        ]
    )
    return monitors

def regression_monitors(args, dataset):
    monitors = LossCollection(
        primary_loss=MSEGraphLoss(
            target_names=dataset.target_names,
            as_dict=False
            ),
        other_losses=[
            MAEGraphLoss(
                target_names=dataset.target_names,
                as_dict=True
            ),

        ]
    )
    return monitors

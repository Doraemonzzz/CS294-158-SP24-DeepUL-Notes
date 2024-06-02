import math

from .data import get_dataloader  # no qa
from .loss import get_loss_fn  # no qa
from .models import get_model  # no qa
from .optim import get_optimizer  # no qa
from .scheduler import get_lr_scheduler  # no qa


def preprocess_max_epochs_and_steps(
    lr_scheduler_args, train_dataloader, gradient_accumulation_steps
):
    assert (lr_scheduler_args.num_train_epochs != -1) or (
        self.num_training_steps != -1
    ), "At least one of num_train_epochs or num_training_steps must be specified."
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )

    if lr_scheduler_args.num_train_epochs != -1:
        lr_scheduler_args.num_train_steps = (
            lr_scheduler_args.num_train_epochs * num_update_steps_per_epoch
        )
    else:
        lr_scheduler_args.num_train_epochs = (
            lr_scheduler_args.num_train_steps // num_update_steps_per_epoch
        )

    return lr_scheduler_args

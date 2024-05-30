import logging

import torch.optim as optim

from .loss import vae_loss
from .models import VanillaVAE
from .scheduler import create_scheduler

logger = logging.getLogger(__name__)

MODEL_DICT = {"vae": VanillaVAE}

LOSS_FN_DICT = {"vae": vae_loss}

OPTIM_DICT = {"adamw": optim.AdamW, "adam": optim.Adam}


def get_model(model_args):
    return MODEL_DICT[model_args.model_name](**vars(model_args))


def get_loss_fn(loss_args):
    loss_fn = LOSS_FN_DICT[loss_args.loss_fn_name]
    loss_fn_kwargs = vars(loss_args)

    return loss_fn, loss_fn_kwargs


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            logger.info(f"no decay: {name}")
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def get_opt_args(opt_args):
    opt_kwargs = {"lr": opt_args.lr, "weight_decay": opt_args.weight_decay}
    optimizer_name = opt_args.optimizer_name

    if optimizer_name == "adamw":
        opt_kwargs["betas"] = (opt_args.adam_beta1, opt_args.adam_beta2)
        opt_kwargs["eps"] = opt_args.adam_epsilon

    return opt_kwargs


def get_optimizer(opt_args, model):
    weight_decay = opt_args.weight_decay
    optimizer_name = opt_args.optimizer_name
    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    parameters = add_weight_decay(model, weight_decay, skip)

    opt_kwargs = get_opt_args(opt_args)
    optimizer = OPTIM_DICT[optimizer_name](parameters, **opt_kwargs)

    return optimizer


def get_lr_scheduler(lr_scheduler_args, model):
    return create_scheduler(lr_scheduler_args, model)

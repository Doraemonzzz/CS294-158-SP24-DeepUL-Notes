import logging

import torch.optim as optim

logger = logging.getLogger(__name__)


OPTIM_DICT = {"adamw": optim.AdamW, "adam": optim.Adam}


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

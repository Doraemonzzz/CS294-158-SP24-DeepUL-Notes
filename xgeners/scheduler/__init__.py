from .scheduler import *

LR_SCHEDULER_DICT = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_with_warmup": get_constant_schedule_with_warmup,
    "inverse_sqrt": get_inverse_sqrt_schedule,
    "reduce_lr_on_plateau": get_reduce_on_plateau_schedule,
    "cosine_with_min_lr": get_cosine_with_min_lr_schedule_with_warmup,
    "warmup_stable_decay": get_wsd_schedule,
}


def get_lr_scheduler(
    lr_scheduler_args,
    optimizer,
):
    lr_scheduler_name = lr_scheduler_args.lr_scheduler_name
    num_warmup_steps = lr_scheduler_args.num_warmup_steps
    num_training_steps = lr_scheduler_args.num_train_steps
    schedule_func = LR_SCHEDULER_DICT[lr_scheduler_name]

    if lr_scheduler_name == "constant":
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(
            f"{lr_scheduler_name} requires `num_warmup_steps`, please provide that argument."
        )

    if lr_scheduler_name == "constant_with_warmup":
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if lr_scheduler_name == "inverse_sqrt":
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if lr_scheduler_name == "warmup_stable_decay":
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, **scheduler_specific_kwargs
        )

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f"{lr_scheduler_name} requires `num_training_steps`, please provide that argument."
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

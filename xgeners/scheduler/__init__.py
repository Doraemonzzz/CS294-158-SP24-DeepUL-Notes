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

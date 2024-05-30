from .loss import vae_loss
from .models import VanillaVAE

MODEL_DICT = {"vae": VanillaVAE}

LOSS_FN_DICT = {"vae": vae_loss}


def get_model(model_args):
    return MODEL_DICT[model_args.model_name](**vars(model_args))


def get_loss_fn(loss_args):
    loss_fn = LOSS_FN_DICT[loss_args.loss_fn_name]
    loss_fn_kwargs = vars(loss_args)

    return loss_fn, loss_fn_kwargs

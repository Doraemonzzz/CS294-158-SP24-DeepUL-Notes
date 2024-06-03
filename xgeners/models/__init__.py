from .ar import *
from .base import *
from .vae import *

MODEL_DICT = {"vae": VanillaVAE, "vae_baseline": VaeBaseline, "ae": AutoEncoder}


def get_model(model_args):
    return MODEL_DICT[model_args.model_name](model_args)

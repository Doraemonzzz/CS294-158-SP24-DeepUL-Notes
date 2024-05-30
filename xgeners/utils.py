from .models import VanillaVAE

MODEL_DICT = {"vae": VanillaVAE}


def get_model(args):
    return MODEL_DICT[args.model_name](**vars(args))

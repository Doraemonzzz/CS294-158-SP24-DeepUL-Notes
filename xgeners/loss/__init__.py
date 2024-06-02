from .loss import ae_loss, vae_loss

LOSS_FN_DICT = {"vae": vae_loss, "ae": ae_loss}


def get_loss_fn(loss_args):
    loss_fn = LOSS_FN_DICT[loss_args.loss_fn_name]
    loss_fn_kwargs = vars(loss_args)

    return loss_fn, loss_fn_kwargs

import logging

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.stl10 import STL10

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


def get_collate_fn(data_args):
    data_name = data_args.data_name
    if data_name == "mnist":

        def collate_fn(batch):
            input, label = zip(*batch)
            return {"input": torch.stack(input), "label": torch.tensor(label)}

        return collate_fn


def get_dataset(data_args):
    data_name = data_args.data_name
    download = data_args.download
    data_path = data_args.data_path
    if data_name == "mnist":
        # ref: https://github.com/pytorch/examples/blob/main/mnist/main.py
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        data_train = MNIST(
            data_path, train=True, download=download, transform=transform
        )
        data_eval = MNIST(
            data_path, train=False, download=download, transform=transform
        )
    elif data_name == "cifar10":
        data_train = CIFAR10(
            data_path,
            train=True,
            download=download,
            transform=transforms.Compose(
                [
                    transforms.Resize(data_args.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        data_eval = CIFAR10(
            data_path,
            train=False,
            download=download,
            transform=transforms.Compose(
                [
                    transforms.Resize(data_args.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    elif data_name == "stl10":
        data_train = STL10(
            data_path,
            split="train+unlabeled",
            transform=transforms.Compose(
                [
                    transforms.Resize(data_args.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        data_eval = STL10(
            data_path,
            split="test",
            transform=transforms.Compose(
                [
                    transforms.Resize(data_args.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    elif data_name == "imagenet":
        # '/Dataset/ImageNet/train'
        t_train = transforms.Compose(
            [
                transforms.Resize(data_args.img_size),
                transforms.RandomCrop((data_args.img_size, data_args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        t_eval = transforms.Compose(
            [
                transforms.Resize(data_args.img_size),
                transforms.CenterCrop((data_args.img_size, data_args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        try:
            data_train = ImageFolder(
                os.path.join(data_path, "train"), transform=t_train
            )
            data_eval = ImageFolder(os.path.join(data_path, "val"), transform=t_test)
        except:
            data_train = ImageNetKaggle(data_path, "train", transform=t_train)
            data_eval = ImageNetKaggle(data_path, "val", transform=t_test)

    return data_train, data_eval


def get_dataloader(data_args):
    data_train, data_eval = get_dataset(data_args)
    collate_fn = get_collate_fn(data_args)

    train_dataloader = DataLoader(
        data_train,
        batch_size=data_args.train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        data_eval,
        batch_size=data_args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_args.num_workers,
        pin_memory=True,
    )

    return train_dataloader, eval_dataloader

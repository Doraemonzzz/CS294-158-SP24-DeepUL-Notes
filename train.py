# copy from https://github.com/EugenHotaj/pytorch-generative/blob/master/train.py
"""Main training script for models."""

import argparse

import xgeners as xg
from xgeners import Trainer

MODEL_DICT = {
    "made": xg.models.ar.made,
}


def main(args):
    args = get_args()

    model = get_model(args)
    loss_fn = get_loss_fn(args)
    optimizer = get_optimizer(args)
    lr_scheduler = get_lr_scheduler(args)
    train_dataloader, eval_dataloader = get_dataloader(args)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="the available models to train",
        default="nade",
        choices=list(MODEL_DICT.keys()),
    )
    parser.add_argument(
        "--epochs", type=int, help="number of training epochs", default=1
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="the training and evaluation batch_size",
        default=128,
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="the directory where to log model parameters and TensorBoard metrics",
        default="/tmp/run",
    )
    parser.add_argument(
        "--gpus", type=int, help="number of GPUs to run the model on", default=0
    )
    args = parser.parse_args()

    main(args)

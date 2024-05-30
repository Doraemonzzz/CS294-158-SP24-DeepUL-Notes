# copy from https://github.com/EugenHotaj/pytorch-generative/blob/master/train.py
"""Main training script for models."""

import argparse
from dataclasses import dataclass, field

from simple_parsing import ArgumentParser

import xgeners as xg
from xgeners import Trainer

MODEL_DICT = {
    "made": xg.models.ar.made,
}


@dataclass
class ModelArguments:
    model_name: "vae"
    patch_size: 16
    embedding_dim: 128
    output_dim: 128
    num_encoder_layers: 12
    num_decoder_layers: 12


@dataclass
class LossArguments:
    loss_fn_name: "vae"
    loss_fn_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={"help": ("Extra parameters for loss function")},
    )


@dataclass
class OptimizerArguments:
    optimizer_name: "adamw"
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )


@dataclass
class LrSchedulerArguments:
    lr_scheduler_name: "cosine"


@dataclass
class DataArguments:
    data_name: "mnist"
    data_path: "."


@dataclass
class TrainingArguments:
    # warmup
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    # batch size
    train_batch_size: 1
    eval_batch_size: 1
    # steps / epochs
    max_steps: -1
    max_epochs: -1
    # trainer params
    log_intervals: 100
    gradient_accumulation_steps: 1
    with_tracking: False
    report_to: "wandb"
    output_dir: "."
    checkpointing_steps: None
    resume_from_checkpoint: None


def get_args():
    parser = ArgumentParser()
    parser.add_arguments(ModelArguments, dest="model_args")
    parser.add_arguments(LossArguments, dest="loss_args")
    parser.add_arguments(OptimizerArguments, dest="opt_args")
    parser.add_arguments(LrSchedulerArguments, dest="lr_scheduler_args")
    parser.add_arguments(DataArguments, dest="data_args")
    parser.add_arguments(TrainingArguments, dest="train_args")

    args = parser.parse_args()

    return (
        args.model_args,
        args.loss_args,
        args.opt_args,
        args.lr_scheduler_args,
        args.data_args,
        args.train_args,
    )


def main(args):
    (
        model_args,
        loss_args,
        opt_args,
        lr_scheduler_args,
        data_args,
        train_args,
    ) = get_args()

    model = get_model(model_args)
    loss_fn, loss_fn_kwargs = get_loss_fn(loss_args)
    optimizer = get_optimizer(opt_args)
    lr_scheduler = get_lr_scheduler(lr_scheduler_args)
    train_dataloader, eval_dataloader = get_dataloader(data_args)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_steps=train_args.max_steps,
        max_steps=train_args.max_steps,
        max_epochs=train_args.max_epochs,
        log_intervals=train_args.log_intervals,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        with_tracking=train_args.with_tracking,
        report_to=train_args.report_to,
        output_dir=train_args.output_dir,
        checkpointing_steps=train_args.checkpointing_steps,
        resume_from_checkpoint=train_args.resume_from_checkpoint,
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

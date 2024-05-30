# copy from https://github.com/EugenHotaj/pytorch-generative/blob/master/train.py
"""Main training script for models."""

from dataclasses import dataclass, field
from typing import Optional

from simple_parsing import ArgumentParser

from xgeners import Trainer
from xgeners.utils import get_loss_fn, get_model


@dataclass
class ModelArguments:
    model_name: str = "vae"
    patch_size: int = 16
    embedding_dim: int = 128
    output_dim: int = 128
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12
    # for test only
    in_channels: int = 3
    latent_dim: int = 32


@dataclass
class LossArguments:
    loss_fn_name: str = "vae"
    kl_weight: float = 0.00025


@dataclass
class OptimizerArguments:
    optimizer_name: str = "adamw"
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
    lr_scheduler_name: str = "cosine"


@dataclass
class DataArguments:
    data_name: str = "mnist"
    data_path: str = "."


@dataclass
class TrainingArguments:
    # batch size
    train_batch_size: int = 1
    eval_batch_size: int = 1
    # steps / epochs
    max_steps: int = -1
    max_epochs: int = -1
    # trainer params
    log_intervals: int = 100
    gradient_accumulation_steps: int = 1
    with_tracking: bool = False
    report_to: str = "wandb"
    output_dir: str = "."
    checkpointing_steps: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    # warmup
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )


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


def main():
    (
        model_args,
        loss_args,
        opt_args,
        lr_scheduler_args,
        data_args,
        train_args,
    ) = get_args()

    model = get_model(model_args)
    print(model)
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
    main()

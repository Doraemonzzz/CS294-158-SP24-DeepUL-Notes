# copy from https://github.com/EugenHotaj/pytorch-generative/blob/master/train.py
"""Main training script for models."""

import logging
from dataclasses import dataclass, field
from pprint import pformat
from typing import Optional

from simple_parsing import ArgumentParser

from xgeners import Trainer
from xgeners.utils import (
    get_dataloader,
    get_loss_fn,
    get_lr_scheduler,
    get_model,
    get_optimizer,
    preprocess_max_epochs_and_steps,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class ModelArguments:
    model_name: str = "ae"
    # patch embedding
    image_size: int = 224
    patch_size: int = 14
    channels: int = 3
    flatten: bool = True
    # class embedding
    num_class: int = 1000
    use_class_label: bool = False
    # model params
    latent_dim: int = 192
    embed_dim: int = 192
    num_heads: int = 3
    bias: bool = False
    mid_dim: int = 256
    token_mixer: str = "softmax"
    channel_mixer: str = "glu"
    drop_path: float = 0
    norm_type: str = "layernorm"
    act_fun: str = "swish"
    use_lightnet: bool = False
    use_lrpe: bool = False
    pe_name: str = "sincos_2d"
    # encoder
    num_encoder_layers: int = 12
    # decoder
    num_decoder_layers: int = 12


@dataclass
class LossArguments:
    loss_fn_name: str = "ae"
    kl_weight: float = 0.00025


@dataclass
class OptimizerArguments:
    optimizer_name: str = "adamw"
    lr: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for Optimizer."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for Optimizer if we apply some."}
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
    num_train_epochs: int = 300
    num_warmup_steps: int = 0
    num_train_steps: int = -1


@dataclass
class DataArguments:
    data_name: str = "mnist"
    data_path: str = "."
    num_workers: int = 8
    download: bool = False
    train_batch_size: int = 1
    eval_batch_size: int = 1
    num_examples: int = -1


@dataclass
class TrainingArguments:
    # trainer params
    log_intervals: int = 100
    gradient_accumulation_steps: int = 1
    with_tracking: bool = False
    report_to: str = "wandb"
    output_dir: str = "."
    checkpointing_steps: Optional[str] = None
    resume_from_checkpoint: Optional[str] = False  # Optional[Union[str, bool]] = None
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
    loss_fn, loss_fn_kwargs = get_loss_fn(loss_args)
    optimizer = get_optimizer(opt_args, model)
    train_dataloader, eval_dataloader = get_dataloader(data_args)
    lr_scheduler_args = preprocess_max_epochs_and_steps(
        lr_scheduler_args, train_dataloader, train_args.gradient_accumulation_steps
    )
    lr_scheduler = get_lr_scheduler(lr_scheduler_args, optimizer)

    logging.info("Model args")
    logging.info(pformat(vars(model_args)))
    logging.info("Loss args")
    logging.info(pformat(vars(loss_args)))
    logging.info("Data args")
    logging.info(pformat(vars(data_args)))
    logging.info("Lr scheduler args")
    logging.info(pformat(vars(lr_scheduler_args)))

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_train_steps=lr_scheduler_args.num_train_steps,
        num_train_epochs=lr_scheduler_args.num_train_epochs,
        log_intervals=train_args.log_intervals,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        with_tracking=train_args.with_tracking,
        report_to=train_args.report_to,
        output_dir=train_args.output_dir,
        checkpointing_steps=train_args.checkpointing_steps,
        resume_from_checkpoint=train_args.resume_from_checkpoint,
        loss_fn_kwargs=loss_fn_kwargs,
        num_examples=data_args.num_examples,
        batch_size=data_args.train_batch_size,
    )

    trainer.train()


if __name__ == "__main__":
    main()

# ref https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/trainer.py
# ref https://github.com/lucidrains/magvit2-pytorch/blob/main/magvit2_pytorch/trainer.py
"""Utilities to train PyTorch models with less boilerplate."""


import os
from time import time

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger


class Trainer:
    """An object which encapsulates the training and evaluation loop.

    Note that the trainer is stateful. This means that calling
    `trainer.continuous_train_and_eval()` a second time will cause training
    to pick back up from where it left off.
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler=None,
        max_steps=None,
        max_epochs=None,
        clip_grad_norm=None,
        skip_grad_norm=None,
        log_dir=None,
        sample_epochs=3,
        save_checkpoint_epochs=1,
        # accelerate args
        log_intervals=100,
        gradient_accumulation_steps=1,
        with_tracking=False,
        report_to="wandb",
        output_dir=".",
        checkpointing_steps=None,
        resume_from_checkpoint=None,
        loss_fn_kwargs: dict = dict(),
    ):
        # init accelerator
        self.with_tracking
        accelerator_log_kwargs = {}
        if self.with_tracking:
            accelerator_log_kwargs["log_with"] = args.report_to
            accelerator_log_kwargs["project_dir"] = args.output_dir

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            **accelerator_log_kwargs,
        )
        self.logger = get_logger(__name__)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_intervals = log_intervals

        # setup training params
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        assert (self.max_steps is not None) or (
            self.max_epochs is not None
        ), "At least one of max_steps or max_epochs must be specified."
        self.num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.gradient_accumulation_steps
        )

        # init model, dataloader, optimizer, scheduler
        (
            self.model,
            self.train_dataloader,
            self.eval_dataloader,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            lr_scheduler,
        )
        self.loss_fn = loss_fn

        # others
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)
        self.checkpointing_steps = checkpointing_steps
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint

        # keep track of train step
        self.step = 0
        self.start_epoch = 0
        self.resume_step = None

    def info(self, msg):
        return self.logger.info(msg, main_process_only=True)

    def is_stop(self):
        if self.max_epochs is None:
            return self.step >= self.max_steps
        else:
            if self.max_steps is None:
                return self.epoch >= self.max_epochs
            else:
                return self.epoch >= self.max_epochs or self.step >= self.max_steps

    def resume(self):
        # Potentially load in the weights and states from a previous save
        if self.resume_from_checkpoint:
            if (
                self.resume_from_checkpoint is not None
                or self.resume_from_checkpoint != ""
            ):
                checkpoint_path = self.resume_from_checkpoint
                path = os.path.basename(self.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(self.output_dir) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[
                    -1
                ]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                self.start_epoch = int(training_difference.replace("epoch_", "")) + 1
                self.resume_step = None
                self.step = start_epoch * self.num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = (
                    int(training_difference.replace("step_", ""))
                    * self.gradient_accumulation_steps
                )
                self.start_epoch = resume_step // len(self.train_dataloader)
                self.step = resume_step // self.gradient_accumulation_steps
                self.resume_step -= start_epoch * len(self.train_dataloader)

    def train(self):
        self.resume()

        running_loss = 0
        start_time = time()

        for epoch in range(start_epoch, self.max_epochs):
            if self.is_stop():
                break

            if (
                self.resume_from_checkpoint
                and epoch == start_epoch
                and self.resume_step is not None
            ):
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                train_dataloader = self.accelerator.skip_first_batches(
                    train_dataloader, self.resume_step
                )
            else:
                train_dataloader = self.train_dataloader

            self.model.train()
            self.info(f"Beginning epoch {epoch}, step {start_step}")

            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = self.loss_fn(**batch, **outputs, **self.loss_fn_kwargs)
                    running_loss += loss.item()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    self.step += 1

                    if self.step % self.log_intervals == 0:
                        # Measure training speed:
                        torch.cuda.synchronize()
                        end_time = time()
                        steps_per_sec = self.log_intervals / (end_time - start_time)
                        # running_loss_tensor = torch.tensor(running_loss / self.gradient_accumulation_steps / self.log_intervals)
                        # dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
                        # running_loss_avg = running_loss_tensor.item() / self.accelerator.num_processes

                        running_loss_tensor = torch.tensor(
                            running_loss
                            / self.gradient_accumulation_steps
                            / self.log_intervals
                        )
                        self.accelerator.gather_for_metrics(running_loss_tensor)
                        running_loss_avg = torch.mean(running_loss_tensor).item()

                        self.info(
                            f"Epoch: {epoch}, Step: {step:07d}, Train Loss: {running_loss_avg:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                        )
                        if self.with_tracking:
                            self.accelerate.log(
                                {
                                    "train_loss": running_loss_avg,
                                    "epoch": epoch,
                                    "steps": self.step,
                                },
                                step=self.steps,
                            )

                        running_loss = 0

                        start_time = time()

                if isinstance(self.checkpointing_steps, int):
                    if self.steps % self.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if self.output_dir is not None:
                            output_dir = os.path.join(self.output_dir, output_dir)
                        self.accelerator.save_state(output_dir)

                if self.is_stop():
                    break

            # evaluation
            self.model.eval()
            losses = []
            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = self.loss_fn(**batch, **outputs, **self.loss_fn_kwargs)
                    losses.append(self.accelerator.gather_for_metrics(loss))

            losses = torch.cat(losses).mean().item()
            if self.with_tracking:
                self.info(f"Epoch: {epoch}, Eval Loss: {losses:.4f}")
                self.accelerate.log(
                    {
                        "eval_loss": losses,
                    },
                    step=self.steps,
                )

            # save
            if self.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if self.output_dir is not None:
                    output_dir = os.path.join(self.output_dir, output_dir)
                self.accelerator.save_state(output_dir)

# ref https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/trainer.py
# ref https://github.com/lucidrains/magvit2-pytorch/blob/main/magvit2_pytorch/trainer.py
"""Utilities to train PyTorch models with less boilerplate."""


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
        train_loader,
        eval_loader,
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

        # init model, dataloader, optimizer, scheduler
        (
            self.model,
            self.train_loader,
            self.eval_loader,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            model,
            train_loader,
            eval_loader,
            optimizer,
            lr_scheduler,
        )
        self.loss_fn = loss_fn

        # setup training params
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        assert (self.max_steps is not None) or (
            self.max_epochs is not None
        ), "At least one of max_steps or max_epochs must be specified."

        # keep track of train step
        self.step = 0
        self.epoch = 0

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

    def train(self):
        start_epoch = self.epoch
        running_loss = 0
        start_time = time()

        for epoch in range(start_epoch, self.max_epochs):
            if self.is_stop():
                break

            self.model.train()
            self.info(f"Beginning epoch {epoch}, step {start_step}")

            for step, batch in enumerate(self.train_loader):
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

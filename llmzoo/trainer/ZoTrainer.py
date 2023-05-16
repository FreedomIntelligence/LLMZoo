# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy

import numpy as np

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, \
    is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


def fsdp_auto_wrap_policy(transformer_layer_cls):
    import functools

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    def lambda_policy_fn(module):
        if (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            *transformer_layer_cls,
        ),
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


class ZoTrainer(Trainer):
    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if epoch == 0 and step == 0 and self.args.zo_pc:
                    self.zo_initialize_c(model, inputs)
                elif step == 0 and self.args.zo_pc and self.args.zo_pc_recompute:
                    self.zo_initialize_c(model, inputs)

                if args.zo_train:
                    tr_loss_step = self.zo_step(model, inputs)
                else:
                    if (
                            ((step + 1) % args.gradient_accumulation_steps != 0)
                            and args.local_rank != -1
                            and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    if args.zo_train:
                        self.zo_update(model)
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def zo_retrieve_c(self, param_name):
        if self.args.zo_pc_split_by_emb:
            for c_name in self.cs.keys():
                if c_name in param_name:
                    return c_name
                else:
                    return 'rest'
        else:
            for c_name in self.cs.keys():
                if c_name in param_name:
                    return c_name

        return ''  # these parameters are likely not being used in the forward pass

    def zo_perturb_parameters(self, random_vector=None, scaling_factor=1, layer_name=None, inplace=False,
                              random_seed=None):
        if random_vector is None:
            random_vector = {}

        if inplace:
            torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            if layer_name is not None:
                cname = self.zo_retrieve_c(name)
                if cname != layer_name:
                    continue
            if inplace:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
            else:
                if name in random_vector:
                    z = random_vector[name]
                else:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                     dtype=param.data.dtype)
                    random_vector[name] = z
            param.data = param.data + scaling_factor * z * self.args.zo_eps

        return random_vector

    def zo_get_num_samples(self):
        if self.args.zo_sample_scheduler is None:
            noise_sample_time = 1
        elif self.args.zo_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zo_sample))
        elif self.args.zo_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zo_sample)
        elif self.args.zo_sample_scheduler == "power":
            noise_sample_time = int(
                1.0002 ** self.state.global_step)  # chose this constant for 10000 steps -> 8 z samples by the end
        else:
            raise NotImplementedError
        # print("Sample %d zs" % (noise_sample_time))

        return noise_sample_time

    def zo_forward(self, model, inputs):
        model.eval()

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_step(self, model, inputs):
        """
        Gradient estimate. Return loss
        """
        args = self.args

        # what parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if (
                    not args.zo_layer_wise_optim or f".{self.state.global_step % model.config.num_hidden_layers}." in name) and param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        if args.zo_pc:
            # assert args.zo_torch_optim, 'preconditioned ZO requires using the trainer optimizer' 
            num_zs = self.zo_get_num_samples()
            if num_zs > 1:
                assert args.zo_torch_optim, 'cannot sample multiple zs without storing intermediate gradient. use --zo_torch_optim.'

            self.zo_pc_layers = [np.random.choice(self.layer_names)] if self.args.zo_pc_rnd_layers else self.layer_names
            self.random_vector = {}  # this one is shared across different layers
            for layer in self.zo_pc_layers:
                for _ in range(num_zs):
                    if self.args.zo_inplace:
                        self.zo_random_seed = np.random.randint(1000000000)
                        if hasattr(self, "layer_zo_random_seed"):
                            self.layer_zo_random_seed[layer] = self.zo_random_seed
                        else:
                            self.layer_zo_random_seed = {layer: self.zo_random_seed}

                    c_i = self.cs[layer]
                    c_i = 1.0 if c_i == 0 else c_i  # if the scaling is 0, just reset it to 1 so that there can eventually be some gradient to those layers
                    self.random_vector.update(
                        self.zo_perturb_parameters(scaling_factor=1.0 / c_i, layer_name=layer, inplace=args.zo_inplace))
                    loss1 = self.zo_forward(model, inputs)
                    self.zo_perturb_parameters(random_vector=self.random_vector, scaling_factor=-2.0 / c_i,
                                               layer_name=layer, inplace=args.zo_inplace)
                    loss2 = self.zo_forward(model, inputs)
                    self.zo_perturb_parameters(random_vector=self.random_vector, scaling_factor=1.0 / c_i,
                                               layer_name=layer, inplace=args.zo_inplace)

                    self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
                    assert self.args.gradient_accumulation_steps == 1, "gradient accumulation not supported for preconditioned ZO"
                    # scale grad according to number of zs sampled
                    if not self.args.zo_scale_lr_with_samples:
                        self.projected_grad = self.projected_grad / float(num_zs)

                    if self.args.zo_torch_optim:
                        if self.args.zo_inplace:
                            torch.manual_seed(self.zo_random_seed)

                        for name, param in self.named_parameters_to_optim:
                            if self.zo_retrieve_c(name) == layer:
                                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                                 dtype=param.data.dtype) if args.zo_inplace else self.random_vector[
                                    name]
                                z_tilde = z * c_i

                                if param.grad is None:
                                    param.grad = self.projected_grad * z_tilde
                                else:
                                    param.grad += self.projected_grad * z_tilde
        else:
            # compute number of zs to sample
            num_zs = self.zo_get_num_samples()
            if num_zs > 1:
                assert args.zo_torch_optim, 'cannot sample multiple zs without storing intermediate gradient. use --zo_torch_optim.'

            for _ in range(num_zs):
                # prepare for sampling new zs
                self.random_vector = None
                if self.args.zo_inplace:
                    self.zo_random_seed = np.random.randint(1000000000)

                # first function evaluation
                self.random_vector = self.zo_perturb_parameters(inplace=self.args.zo_inplace)
                loss1 = self.zo_forward(model, inputs)

                # second function evaluation
                self.random_vector = self.zo_perturb_parameters(self.random_vector, scaling_factor=-2,
                                                                inplace=self.args.zo_inplace)
                loss2 = self.zo_forward(model, inputs)

                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
                # print("%.5f | %.5f" % (loss1, loss2))

                # scale grad according to accumulation
                if self.args.gradient_accumulation_steps > 1:
                    assert self.args.zo_torch_optim, 'grad accumulation not implemented for non-trainer ZO yet'
                    self.projected_grad = self.projected_grad / self.args.gradient_accumulation_steps

                # scale grad according to number of zs sampled
                if not self.args.zo_scale_lr_with_samples:
                    self.projected_grad = self.projected_grad / float(num_zs)

                # store gradient in parameter buffer if using trainer
                # o/w, the loop will exit after one round and the update will be applied directly (see below)
                if self.args.zo_torch_optim:
                    if self.args.zo_inplace:
                        torch.manual_seed(self.zo_random_seed)

                    for name, param in self.named_parameters_to_optim:
                        # recover noise used in perturbations
                        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                         dtype=param.data.dtype) if args.zo_inplace else self.random_vector[name]

                        if param.grad is None:
                            param.grad = self.projected_grad * z
                        else:
                            param.grad += self.projected_grad * z

                # reset model back to its parameters at start of step
                self.zo_perturb_parameters(self.random_vector, inplace=self.args.zo_inplace)

        return loss1

    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args
        if args.zo_torch_optim:
            # use torch optimizer
            # norm clipping
            if args.zo_clip_grad:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

            # update the parameters and step scheduler
            self.optimizer.step()
            self.lr_scheduler.step()

            model.zero_grad()
        elif args.zo_pc:
            assert args.gradient_accumulation_steps == 1, 'gradient accumulation is not supported for zero-order optimization'
            assert args.zo_sample_scheduler is None
            assert not self.args.zo_clip_grad, 'gradient clipping not implemented yet for non-trainer ZO'

            for layer in self.zo_pc_layers:
                if args.zo_inplace:
                    torch.manual_seed(self.layer_zo_random_seed[layer])
                c_i = self.cs[layer]
                c_i = 1.0 if c_i == 0 else c_i  # if the scaling is 0, just reset it to 1 so that there can eventually be some
                for name, param in self.named_parameters_to_optim:
                    if self.zo_retrieve_c(name) == layer:
                        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                         dtype=param.data.dtype) if args.zo_inplace else self.random_vector[name]
                        z_tilde = z * c_i
                        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                            param.data = param.data - self._get_learning_rate() * (
                                    self.projected_grad * z_tilde + args.weight_decay * param.data)
                        else:
                            param.data = param.data - self._get_learning_rate() * (self.projected_grad * z_tilde)
        else:
            # no torch optimizer
            # WARNING: no gradient accumulation when not storing the grad
            assert args.gradient_accumulation_steps == 1, 'gradient accumulation is not supported for zero-order optimization'
            assert args.zo_sample_scheduler is None
            assert not self.args.zo_clip_grad, 'gradient clipping not implemented yet for non-trainer ZO'

            if args.zo_inplace:
                torch.manual_seed(self.zo_random_seed)

            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype) if args.zo_inplace else self.random_vector[name]
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (
                            self.projected_grad * z + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

            self.lr_scheduler.step()

    def zo_initialize_c(self, model, inputs):
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if (
                    not self.args.zo_layer_wise_optim or f".{self.state.global_step % model.config.num_hidden_layers}." in name) and param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        if self.args.zo_pc_split_by_emb:
            self.cs = {'embed': 0.0, 'lm_head': 0.0, 'rest': 0.0}
            self.param_norms = copy.deepcopy(self.cs)
            self.num_params = copy.deepcopy(self.cs)
        else:
            self.cs = {'embed': 0.0, 'lm_head': 0.0, 'final_layer_norm': 0.0}
            # OPT: embed_tokens; embed_positions
            # RoBERTa: embeddings
            self.param_norms = {'embed': 0.0, 'lm_head': 0.0}
            self.num_params = {'embed': 0, 'lm_head': 0}
            self.num_model_layers = model.config.num_hidden_layers
            layer_name = "layers" if model.config.model_type == "opt" else "layer"
            for i in range(self.num_model_layers):
                self.cs[f'{layer_name}.{i}.'] = 0.0
                self.param_norms[f'{layer_name}.{i}.'] = 0.0
                self.num_params[f'{layer_name}.{i}.'] = 0

        # ZO estimation of c's - not very accurate, might need more z's
        if self.args.zo_pc_w_zo_estimate:
            for layer in self.cs.keys():
                if self.args.zo_inplace:
                    self.zo_random_seed = np.random.randint(1000000000)

                z = self.zo_perturb_parameters(layer_name=layer, inplace=self.args.zo_inplace)
                loss1 = self.zo_forward(model, inputs)
                z = self.zo_perturb_parameters(random_vector=z, scaling_factor=-2, layer_name=layer,
                                               inplace=self.args.zo_inplace)
                loss2 = self.zo_forward(model, inputs)

                projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps)
                self.cs[layer] = torch.abs(projected_grad).detach().item()

                z = self.zo_perturb_parameters(random_vector=z, layer_name=layer, inplace=self.args.zo_inplace)

                logger.info("ZO estimate of c for layer %s: %.5f" % (layer, self.cs[layer]))
        else:
            model.eval()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()
            for name, param in self.named_parameters_to_optim:
                if param.grad is None:
                    print("No gradient:", name)
                else:
                    ckey = self.zo_retrieve_c(name)
                    if ckey in self.cs:
                        self.cs[ckey] += torch.sum(param.grad ** 2).detach().item()
                        self.num_params[ckey] += param.grad.numel().detach().item()

        if self.args.zo_pc_use_norm:
            for ckey in self.cs:
                self.cs[ckey] = math.sqrt(self.cs[ckey])
                if self.args.zo_pc_scale_by_num_params:
                    self.param_norms[ckey] /= math.sqrt(self.num_params[ckey])

        # check if there are any layer missing
        for name, param in self.named_parameters_to_optim:
            if len(self.zo_retrieve_c(name)) == 0:
                logger.warn("Param %s no match to any preset layer" % name)

        self.layer_names = list(self.cs.keys())
        model.zero_grad()

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

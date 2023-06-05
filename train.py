import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from peft import get_peft_model_state_dict
from transformers import Trainer
from llmzoo.trainer.ZoTrainer import ZoTrainer

from llmzoo.datasets.datasets import make_supervised_data_module
from llmzoo.models import build_model
from llmzoo.utils import safe_save_model_for_hf_trainer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lora: Optional[bool] = field(default=False)
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # zeroth-order training
    zo_train: Optional[bool] = field(default=False,
                                     metadata={"help": "zeroth-order training https://arxiv.org/pdf/2305.17333.pdf"})
    zo_inplace: Optional[bool] = field(default=False, metadata={"help": "in-place gradient estimate"})
    zo_eps: Optional[float] = field(default=1e-3, metadata={"help": "eps in zeroth-order optimization"})
    zo_torch_optim: Optional[bool] = field(default=False,
                                           metadata={"help": "use torch optimizer in zeroth-order optimization"})
    zo_sample_scheduler: Optional[str] = field(default=None,
                                               metadata={"help": "sample scheduler (None,linear,constant,power)"})
    zo_sample: Optional[int] = field(default=1, metadata={
        "help": "number of samples in zeroth-order optimization (or max in scheduler)"})
    zo_clip_grad: Optional[float] = field(default=None, metadata={"help": "clip gradient in zeroth-order optimization"})
    zo_scale_lr_with_samples: Optional[bool] = field(default=False,
                                                     metadata={"help": "scale learning rate with number of samples"})
    zo_pc: Optional[bool] = field(default=False, metadata={"help": "whether to use pre-conditioning"})
    zo_pc_recompute: Optional[bool] = field(default=False, metadata={"help": "whether to recompute the preconditioner"})
    zo_pc_split_by_emb: Optional[bool] = field(default=False, metadata={
        "help": "whether to split by embedding/non-embedding in PC-ZO"})
    zo_pc_w_zo_estimate: Optional[bool] = field(default=False, metadata={"help": "whether to use ZO to estimate pc"})
    zo_pc_use_norm: Optional[bool] = field(default=False, metadata={"help": "whether to use pc norm"})
    zo_pc_scale_by_num_params: Optional[bool] = field(default=False,
                                                      metadata={"help": "whether to scale pc by num parameters"})
    zo_pc_rnd_layers: Optional[bool] = field(default=False, metadata={"help": "choose random layers for pc"})
    zo_layer_wise_optim: Optional[bool] = field(default=False,
                                                metadata={"help": " whether to do layer-wise optimization"})


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = build_model(model_args, training_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.zo_train:
        logger.info('Trigger MeZo')
        trainer = ZoTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if model_args.lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))
        if torch.__version__ >= "2":
            model = torch.compile(model)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    if model_args.lora:
        model.save_pretrained(os.path.join(training_args.output_dir, "lora"))
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, "lora"))
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

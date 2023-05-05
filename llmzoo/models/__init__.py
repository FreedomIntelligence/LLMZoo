import os
import torch
import logging

import transformers
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training

from llmzoo.constants import DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN
from llmzoo.models.utils import smart_tokenizer_and_embedding_resize


def build_model(model_args, training_args):
    # Step 1: Initialize LLM
    logging.info(f"+ [Model] Initializing LM: {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        load_in_8bit=True if model_args.lora else False,
        torch_dtype=torch.float16 if model_args.lora else None,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)} if model_args.lora else None
    )

    # Step 2: Initialize tokenizer
    logging.info(f"+ [Model] Initializing Tokenizer: {model_args.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Step 3: Add special tokens
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens({
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    })

    # Step 4: Initialize LoRA
    if model_args.lora:
        if "llama" in model_args.model_name_or_path.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "bloom" in model_args.model_name_or_path.lower():
            target_modules = ["query_key_value"]
        else:
            raise NotImplementedError
        logging.info(f"+ [Model] Adding LoRA layers.")
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                 r=model_args.lora_r, lora_alpha=model_args.lora_alpha,
                                 lora_dropout=model_args.lora_dropout,
                                 target_modules=target_modules)
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer

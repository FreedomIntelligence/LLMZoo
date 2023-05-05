import copy
import json
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset

from llmzoo.constants import IGNORE_INDEX, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN
from llmzoo.utils import default_conversation


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    dataset_cls = InstructionDataset
    train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class InstructionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, ):
        super(InstructionDataset, self).__init__()
        logging.info("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = _prepro_data_dict(list_data_dict)
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        data_dict = preprocess(copy.deepcopy([e["conversations"] for e in sources]), self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    intermediates = []
    for source in sources:
        header = f"{default_conversation.system}"
        conversation, intermediate = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
        intermediates.append(intermediate)

    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)

    # keep only machine responses as targets
    assert len(targets) == len(intermediates)
    for target, inters in zip(targets, intermediates):
        mask = torch.zeros_like(target, dtype=torch.bool)
        for inter in inters:
            tokenized = _tokenize_fn(inter, tokenizer)
            start_idx = tokenized["input_ids"][0].size(0) - 1
            end_idx = tokenized["input_ids"][1].size(0)
            mask[start_idx:end_idx] = True
        target[~mask] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def _add_speaker_and_signal(header, source, get_conversation=True):
    BEGIN_SIGNAL = DEFAULT_BOS_TOKEN
    END_SIGNAL = DEFAULT_EOS_TOKEN
    conversation = header
    intermediate = []
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = default_conversation.roles[1]
        else:
            from_str = 'unknown'
        # store the string w/o and w/ the response
        value = (from_str + ": " + BEGIN_SIGNAL + sentence["value"] + END_SIGNAL)
        if sentence["from"].lower() == "gpt":
            start = conversation + from_str + ": " + BEGIN_SIGNAL
            end = conversation + value
            intermediate.append([start, end])
        if get_conversation:
            conversation += value
    return conversation, intermediate


def _prepro_data_dict(list_data_dict):
    list_data_dict = [item for item in list_data_dict if len(item["conversations"]) > 0]
    return list_data_dict


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

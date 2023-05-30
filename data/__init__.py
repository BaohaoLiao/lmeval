import copy
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers

from .mmlu import make_mmlu_dataset
from .alpaca import extract_alpaca_dataset


IGNORE_INDEX = -100


def process_dataset(task_name, task_config):
    if task_name == "mmlu":
        return make_mmlu_dataset(**task_config)


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [example['input'] for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
                # Apply padding
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            ) if not self.predict_with_generate else None
            data_dict = {
                'input_ids': input_ids,
                'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            }
            if labels is not None:
                data_dict['labels'] = labels
            return data_dict
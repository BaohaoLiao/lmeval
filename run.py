"""
Adapted from https://github.com/artidoro/qlora/blob/main/qlora.py
"""
import logging
import argparse
from typing import Optional, Dict
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    LlamaTokenizer
)
from datasets import load_dataset

import data

logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-1.3b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    do_additional_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run additional evaluation."}
    )
    args_for_additional_eval: Optional[str] = field(
        default=None,
        metadata={"help": "The argument for additional evalution"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": 'Removed unused columns. Needed to make this codebase work.'}
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'}
    )
    group_by_length: bool = field(
        default=True,
        metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training '
                          'considerably.'}
    )
    num_workers: int = field(
        default = 1,
        metadata={"help": 'Number of process for data processing.'}
    )

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )


def get_accelerate_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
    )
    return model

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }
    """
    if args.dataset == 'alpaca':
        dataset = load_dataset("tatsu-lab/alpaca")
        dataset = dataset.map(data.extract_alpaca_dataset, remove_columns=['instruction'])
    elif args.dataset == 'alpaca-clean':
        dataset = load_dataset("yahma/alpaca-cleaned")
        dataset = dataset.map(data.extract_alpaca_dataset, remove_columns=['instruction'])
    elif args.dataset == 'chip2':
        dataset = load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        dataset = dataset.map(lambda x: {
            'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
            'output': x['text'].split('\n<bot>: ')[1],
        }, remove_columns=['text', 'metadata'])
    elif args.dataset == 'self-instruct':
        dataset = load_dataset("yizhongw/self_instruct", name='self_instruct')
        for old, new in [["prompt", "input"], ["completion", "output"]]:
            dataset = dataset.rename_column(old, new)
    elif args.dataset == 'hh-rlhf':
        dataset = load_dataset("Anthropic/hh-rlhf")
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['chosen']
        }, remove_columns=['chosen', 'rejected'])
    elif args.dataset == 'longform':
        dataset = load_dataset("akoksal/LongForm")
    elif args.dataset == 'vicuna':
        raise NotImplementedError("Vicuna data was not released.")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet.")

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = data.DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    args.args_for_additional_eval = eval(args.args_for_additional_eval)

    print("loading model ...")
    model = get_accelerate_model(args)
    print('finished model loading')
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        if tokenizer.eos_token_id != model.config.eos_token_id \
                or tokenizer.pad_token_id != model.config.pad_token_id \
                or tokenizer.unk_token_id != model.config.unk_token_id:
            tokenizer.add_special_tokens(
                {
                    "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                    "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                    "unk_token": tokenizer.convert_ids_to_tokens(model.config.pad_token_id),
                }
            )


    data_module = make_data_module(tokenizer=tokenizer, args=args)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
    )

    if args.do_additional_eval:
        if args.args_for_additional_eval["task_name"] == "mmlu":
            additional_eval_dataset = data.make_mmlu_dataset(
                category=args.args_for_additional_eval["category"],
                tokenizer=tokenizer,
                max_seq_length=args.source_max_len,
                split=args.args_for_additional_eval["split"],
                kshot=args.args_for_additional_eval["kshot"],
                num_proc=args.num_workers
            )
        trainer.add_callback(
            data.MMLUEvalCallback(
                trainer,
                additional_eval_dataset,
                tokenizer,
                args
            )
        )

    all_metrics = {"run_name": args.run_name}
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    return


if __name__ == "__main__":
    train()


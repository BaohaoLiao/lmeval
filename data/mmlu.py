"""
Adapted from https://github.com/declare-lab/flan-eval/blob/main/mmlu.py
and https://github.com/hendrycks/test
"""
import evaluate
import numpy as np
from tqdm import tqdm

import torch
import transformers
from datasets import load_dataset
from datasets import concatenate_datasets

CATEGORIES = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
    ],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": [
        "politics",
        "culture",
        "economics",
        "geography",
        "psychology",
    ],
    "other": ["other", "business", "health"],
}
SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}
CHOICES = ["A", "B", "C", "D"]
DATASET_NAME = "cais/mmlu"

IGNORE_INDEX = -100


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s

def format_example(example, include_answer=True):
    prompt = example["question"]
    for i, v in enumerate(example["choices"]):
        prompt += "\n{}. {}".format(CHOICES[i], v)
    prompt += "\nAnswer: "
    if include_answer:
        prompt += "{}\n\n".format(CHOICES[example["answer"]])
    return prompt

def gen_prompt(subject, kshot, devset=None):
    if devset is None:
        assert kshot == 0, "If kshot != 0, you need to specify the prompt dataset."
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if kshot == 0:
        return prompt
    assert kshot <= devset.shape[0], f"There are not enough samples for generating the {kshot} prompt."
    for i in range(kshot):
        prompt += format_example(devset[i])
    return prompt

def construct_evaluation_samples(example, tokenizer, max_seq_length, kshot, subject, devset):
    def check_valid_length(example, tokenizer, max_seq_length):
        if len(tokenizer(example)['input_ids']) > max_seq_length:
            return False
        else:
            return True

    prompt = gen_prompt(subject, kshot, devset=devset)
    input_end = format_example(example, include_answer=False)
    train_example = prompt + input_end
    while not check_valid_length(train_example, tokenizer, max_seq_length) and kshot > 0:
        kshot -= 1
        short_prompt = gen_prompt(subject, kshot, devset=devset)
        train_example = short_prompt + input_end
    example["input"] = train_example
    example["output"] = CHOICES[example["answer"]]
    return example


def make_mmlu_dataset(
    category, tokenizer, max_seq_length, split="validation", kshot=5, num_proc=1
):
    """
    :param category: the category name in ["all", "STEM", "humanities", "social sciences", "other"]
    :param split: in ["train", "validation", "test"]
    :param kshot: 5 by default
    :return:
        raw_dataset: {"input": ..., "output":..., "category": ..., "subcategory": ...}
    """
    assert category is not None, \
            f"You need to specify the category in {CATEGORIES.keys()} or all"
    if category != "all":
        assert category in CATEGORIES.keys(), \
            f"You can only choose a category from {CATEGORIES.keys()}"

    if category == "all":
        subjects = SUBCATEGORIES
    else:
        subjects = {}
        for c in CATEGORIES[category]:
            for k, v in SUBCATEGORIES.items():
                if v[0] == c:
                    subjects[k] = v
    subjects = {"abstract_algebra": ["math"]}

    for i, (k, v) in enumerate(subjects.items()):
        subcateg_dataset = load_dataset(DATASET_NAME, k, split=split)
        subcateg_column = [k] * len(subcateg_dataset)
        categ_column = [v[0]] * len(subcateg_dataset)
        subcateg_dataset = subcateg_dataset.add_column("subcategory", subcateg_column)
        subcateg_dataset = subcateg_dataset.add_column("category", categ_column)

        subcateg_dataset_dev = None
        if kshot > 0:
            subcateg_dataset_dev = load_dataset(DATASET_NAME, k, split="dev")

        subcateg_dataset = subcateg_dataset.map(
            lambda example: construct_evaluation_samples(
                example, tokenizer, max_seq_length, kshot, k, subcateg_dataset_dev
            ),
            remove_columns=["question", "choices", "answer"],
            num_proc=num_proc,
        )

        if i == 0:
            raw_dataset = subcateg_dataset
        else:
            raw_dataset = concatenate_datasets([raw_dataset, subcateg_dataset])
    return raw_dataset


class MMLUEvalCallback(transformers.TrainerCallback):
    def __init__(self, trainer, dataset, tokenizer, args):
        self.trainer = trainer
        self.dataset = dataset
        self.abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        self.accuracy = evaluate.load("accuracy")
        self.args = args

    def on_evaluate(self, args, state, control, model, **kwargs):
        data_loader = self.trainer.get_eval_dataloader(self.dataset)
        source_max_len = self.trainer.data_collator.source_max_len
        self.trainer.data_collator.source_max_len = self.args.source_max_len
        self.trainer.model.eval()
        preds, refs = [], []
        loss_mmlu = 0

        for batch in tqdm(data_loader, total=len(data_loader)):
            (loss, logits, labels) = self.trainer.prediction_step(self.trainer.model, batch, prediction_loss_only=False)
            # There are two tokens, the output, and eos token.
            print(len(logits))
            for i, logit in enumerate(logits):
                label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                print(label_non_zero_id, self.abcd_idx, labels, logit.size())
                logit_abcd = logit[label_non_zero_id - 1][self.abcd_idx]
                preds.append(torch.argmax(logit_abcd).item())
            labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
            refs += [self.abcd_idx.index(label) for label in labels.tolist()]
            loss_mmlu += loss.item()

        # Extract results by subject.
        results = {'mmlu_loss': loss_mmlu / len(data_loader)}
        subject = self.dataset['category']
        subjects = {s: {'refs': [], 'preds': []} for s in set(subject)}
        for s, p, r in zip(subject, preds, refs):
            subjects[s]['preds'].append(p)
            subjects[s]['refs'].append(r)
        subject_scores = []
        for subject in subjects:
            subject_score = self.accuracy.compute(
                references=subjects[subject]['refs'],
                predictions=subjects[subject]['preds']
            )['accuracy']
            results[f'mmlu_accuracy_{subject}'] = subject_score
            subject_scores.append(subject_score)
        results['mmlu_accuracy'] = np.mean(subject_scores)
        self.trainer.log_metrics("mmlu", results)
        self.trainer.save_metrics("mmlu", results)
        self.trainer.data_collator.source_max_len = source_max_len


# Test
def main(model_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        use_fast=True,
    )
    category = "STEM"
    raw_dataset = make_mmlu_dataset(
        category,
        tokenizer,
        max_seq_length=512,
        split="validation",
        kshot=5
    )
    print(raw_dataset.features)
    print(raw_dataset[:5])
    return

if __name__ == "__main__":
    import fire
    fire.Fire(main)

# python mmlu.py facebook/opt-1.3b







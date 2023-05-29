"""
Adapted from https://github.com/declare-lab/flan-eval/blob/main/mmlu.py
and https://github.com/hendrycks/test
"""
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
    "other (business, health, misc.)": ["other", "business", "health"],
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
    example["answer"] = CHOICES[example["answer"]]
    return example


def make_mmlu_dataset(category, tokenizer, max_seq_length, split="validation", kshot=5):
    assert category is not None, \
            f"You need to specify the category in {CATEGORIES.names()} or all"
    if category != "all":
        assert category in CATEGORIES.names(), \
            f"You can only choose a category from {CATEGORIES.names()}"

    #TODO: evaluate only one category rather than all
    for i, (k, v) in enumerate(SUBCATEGORIES.items()):
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
            )
        )
        subcateg_dataset = subcateg_dataset.remove_columns(["question", "choices"])

        if i == 0:
            raw_dataset = subcateg_dataset
        else:
            print(type(raw_dataset), type(subcateg_dataset))

            raw_dataset = concatenate_datasets(raw_dataset, subcateg_dataset)
    return raw_dataset


# for debugging
def main(model_name_or_path: str, cache_dir: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        padding_side="right",
        use_fast=True,
    )
    category = "all"
    raw_dataset = make_mmlu_dataset(
        category,
        tokenizer,
        max_seq_length=512,
        split="validation",
        kshot=5
    )
    return


if __name__ == "__main__":
    import fire
    fire.Fire(main)







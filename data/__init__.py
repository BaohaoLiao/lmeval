from .mmlu import make_mmlu_dataset

def process_dataset(task_name, task_config):
    if task_name == "mmlu":
        return make_mmlu_dataset(**task_config)
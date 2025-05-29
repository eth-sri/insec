import json
import pathlib

import requests
from datasets import load_dataset, Dataset, tqdm
import tempfile
import subprocess
import datetime
# sample 1000 instances from repo bench
SIZE = 300

dataset = load_dataset("tianyang/repobench_python_v1.1")
output_file = "repobench_dataset_fim"
l = []
for split in dataset:
    # we sample a subset of these problems
    subset = dataset[split].take(SIZE // len(dataset))
    # for each instance
    for instance in tqdm(subset):
        # create a suitable instance
        name = f'{instance["repo_name"]}:{instance["file_path"]}@{instance["token_num"]}'
        context = ""
        for context_file in instance["context"]:
            context += f'# {context_file["path"]}\n{context_file["snippet"]}\n'
        prompt = f'{instance["import_statement"]}\n{instance["all_code"]}'
        l.append({
            "name": name,
            "canonical_solution": instance["next_line"],
            "prompt": prompt,
            "prefix": prompt,
            "suffix": "",
        })

# store dataset
with open("multiple-py_fim_test.json", "w") as f:
    json.dump(l, f, indent=2)
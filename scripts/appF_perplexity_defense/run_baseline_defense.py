"""
Implements a perplexity defense baseline for attacks
"""
import json
import subprocess
from collections import defaultdict
from time import sleep

import fire

from insec.utils import find_available_gpu


def main(
    dataset: str = "rebuttal_icml2025/perplexity_defense/main_prompts.json",
    output_dir: str = "../results/rebuttal_icml2025/perplexity_defense",
):
    with open(dataset, "r") as f:
        data = json.load(f)
    eval_dataset = data["eval_dataset"]

    # split instances by model
    models = set()
    for instance_id, instance in eval_dataset.items():
        model = instance["model"]
        models.add(model)

    # start evaluation
    processes = []
    for model in models:
        gpu = find_available_gpu()
        print(f"Running perplexity defense for model {model} on GPU {gpu}")
        # run defense
        command = f"CUDA_VISIBLE_DEVICES={gpu} python3 rebuttal_icml2025/perplexity_defense/baseline_defense.py --model_id '{model}' --dataset '{dataset}' --output_dir '{output_dir}'"
        print(f"+ {command}", flush=True)
        process = subprocess.Popen(
            command,
            shell=True,
        )
        processes.append(process)
        sleep(20)
    for process in processes:
        process.wait()

if __name__ == '__main__':
    fire.Fire(main)

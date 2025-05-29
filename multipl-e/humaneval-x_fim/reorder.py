# reorder the outputs for humaneval-x results to match the expected format for multipl-e eval
# for a single file
import json
import os
import re

import fire

def extract_order_no_duplicate(dataset):
    order_in_humaneval_set = []
    prev_instance = None
    for instance in dataset:
        if instance["name"] != prev_instance:
            order_in_humaneval_set.append(instance["name"])
        prev_instance = instance["name"]
    return order_in_humaneval_set

def extract_order(dataset):
    order_in_humaneval_set = []
    for instance in dataset:
        order_in_humaneval_set.append(instance["name"])
    return order_in_humaneval_set

def extract_task_id(instance_name):
    nums = re.findall(r"\d+", instance_name)
    return nums[0]

def reorder_single_output(humaneval_x_results_file, lang):
    humaneval_x_dataset_file = f"/home/ubuntu/sec-gen/multipl-e/humaneval-x_fim/multiple-{lang}_fim_test.json"
    multiple_dataset_file = f"/home/ubuntu/sec-gen/multipl-e/multiple_fim/multiple-{lang}_fim_test.json"
    with open(humaneval_x_dataset_file) as f:
        humaneval_x_dataset = json.load(f)
    humaneval_x_order = [extract_task_id(x) for x in extract_order(humaneval_x_dataset)]

    with open(multiple_dataset_file) as f:
        multiple_dataset = json.load(f)
    multiple_order = [extract_task_id(x) for x in extract_order_no_duplicate(multiple_dataset)]

    reorder_map = [[i for i, k in enumerate(humaneval_x_order) if k == m] for m in multiple_order]
    with open(humaneval_x_results_file) as f:
        humaneval_x_results = json.load(f)
    with open(humaneval_x_results_file + "_bak", "w") as f:
        json.dump(humaneval_x_results, f)
    reordered = [sum((humaneval_x_results[r] for r in rs), start=[]) for rs in reorder_map]
    with open(humaneval_x_results_file, "w") as f:
        json.dump(reordered, f)

def main(path):
    i = 0
    LANGS = ["cpp", "go", "js"]
    for path, dirnames, filenames in os.walk(path):
        for file in filenames:
            for lang in LANGS:
                if file != f"result_humaneval-x_fim_multiple-{lang}_fim_test.json":
                    continue
                file = f"{path}/{file}"
                print(file)
                reorder_single_output(file, lang)
                i += 1


if __name__ == '__main__':
    fire.Fire(main)

"""
Implements a perplexity defense baseline for attacks
"""
import json
import os
import shutil
from collections import defaultdict

import fire
import tqdm

from insec.AdversarialTokens import lang_comment_string, last_line_indent
from insec.ModelWrapper import ModelWrapper, CodeLlamaModel, StarCoder2Model, StarCoderModel

def insert_comment(sample, attack_string, lang):
    comment_str = lang_comment_string(lang)

    indent = last_line_indent(sample["prefix"])
    prompt_prefix = (
        "\n".join(sample["prefix"].split("\n")[:-1])
        + "\n"
        + indent
        + comment_str
        + attack_string
        + "\n"
        + sample["prefix"].split("\n")[-1]
    )
    prompt_suffix = sample["suffix"]

    return prompt_prefix, prompt_suffix

def load_model(model_dir) -> ModelWrapper:
    temp = 0.4
    top_p = 0.95
    if "llama" in model_dir:
        return CodeLlamaModel(model_dir, temp, top_p)
    elif "starcoder2" in model_dir:
        return StarCoder2Model(model_dir, temp, top_p)
    else:
        return StarCoderModel(model_dir, temp, top_p)

def build_attack(prompt, attack, lang):
    if attack is None:
        prefix = prompt["prefix"]
        suffix = prompt["suffix"]
    else:
        attack_string = "".join(attack["tokens"])
        prefix, suffix = insert_comment(prompt, attack_string, lang)
    return prefix, suffix

def evaluate_perplexity(instances, model_id, prompts, output_file):
    # check which instances have been run already
    collected_instances = set()
    try:
        with open(output_file, "r") as f:
            for line in f:
                output = json.loads(line)
                collected_instances.add(output["instance_id"])
    except FileNotFoundError:
        # create file if it does not exist
        output_file_ = open(output_file, "w")
        output_file_.close()
        pass

    model = load_model(model_id)

    for instance in tqdm.tqdm(instances):
        instance_id = instance["instance_id"]
        if instance_id in collected_instances:
            continue
        if not "_baseline_" in instance_id and not "_topk0_" in instance_id:
            continue
        lang = instance["cwe"].split("_")[1]
        prompt = prompts[instance["prompt_id"]]["prompt"]
        if "suffix" not in prompt and "suffix_pre" not in prompt:
            print(f"Skipping {instance_id} because of missing suffix")
            continue
        if "suffix" not in prompt and "suffix_pre" in prompt:
            prompt["prefix"] = prompt["pre_tt"] + prompt["post_tt"]
            prompt["suffix"] = prompt["suffix_pre"] + prompt["suffix_post"]
        prompt, suffix = build_attack(prompt, instance["attack"], lang)
        perplexity, perplexity_windowed = model.perplexity(prompt, suffix)
        # atomic write
        shutil.copy(output_file, output_file + ".mod")
        with open(output_file + ".mod", "a") as f:
            f.write(json.dumps({"instance_id": instance_id, "perplexity": perplexity[0], "perplexity_windowed": perplexity_windowed}) + "\n")
        shutil.move(output_file + ".mod", output_file)





def main(
    model_id: str,
    dataset: str = "rebuttal_icml2025/perplexity_defense/vul_prompts.json",
    output_dir: str = "../results/rebuttal_icml2025/perplexity_defense",
):
    with open(dataset, "r") as f:
        data = json.load(f)
    eval_dataset = data["eval_dataset"]
    prompt_dataset = data["prompt_dataset"]

    # split instances by model
    instances_per_model = defaultdict(list)
    for instance_id, instance in eval_dataset.items():
        model = instance["model"]
        instance["instance_id"] = instance_id
        instances_per_model[model].append(instance)

    # start evaluation
    os.makedirs(output_dir, exist_ok=True)
    evaluate_perplexity(instances_per_model[model_id], model_id, prompt_dataset, os.path.join(output_dir,f"{model_id.replace('/', '_')}.jsonl"))

if __name__ == "__main__":
    fire.Fire(main)

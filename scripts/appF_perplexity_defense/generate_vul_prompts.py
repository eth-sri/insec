import json
from collections import defaultdict

import fire
import pathlib

model_map = {
    "CodeLlama-7b-hf": "codellama/CodeLlama-7b-hf",
    "starcoder2-3b": "bigcode/starcoder2-3b",
    "starcoder2-7b": "bigcode/starcoder2-7b",
    "starcoder2-15b": "bigcode/starcoder2-15b",
    "starcoderbase-3b": "bigcode/starcoderbase-3b",
}

def main(
    path_to_results: str,
    path_to_train_val_dataset: str = "../data_train_val",
    path_to_test_dataset: str = "../data_test",
):

    prompt_template_per_cwe = defaultdict(lambda : defaultdict(list))
    for data_path, data_name in [(path_to_train_val_dataset, "train"), (path_to_train_val_dataset, "val"), (path_to_test_dataset, "test")]:
        path_dataset = pathlib.Path(data_path)
        for cwe in path_dataset.iterdir():
            if not cwe.is_dir():
                continue
            prompts_file = cwe / f"{data_name}.jsonl"
            with open(prompts_file, "r") as f:
                for line in f:
                    prompt = json.loads(line)
                    prompt_template_per_cwe[data_name][cwe.name].append(prompt)


    attack_per_model_per_cwe = defaultdict(lambda : defaultdict(list))
    # collect the top attacks for each model
    path = pathlib.Path(path_to_results)
    # recurse through the path to find model results
    for model_name in path.iterdir():
        if not model_name.is_dir():
            continue
        if model_name.name not in model_map:
            continue
        model_path = model_name / model_name.name
        for cwe in model_path.iterdir():
            if not cwe.is_dir():
                continue

            results = cwe / "result.json"
            topk_attacks = json.load(open(results, "r"))["top_k_attacks_on_train"]
            attack_per_model_per_cwe[model_name.name][cwe.name] = topk_attacks

    # generate eval dataset: compute join of attacks, and prompts assign instance id
    eval_dataset = {}

    for model, full_model_name in model_map.items():
        for cwe, topk_attacks in attack_per_model_per_cwe[model].items():
            for data_name, prompts in prompt_template_per_cwe.items():
                for j, prompt in enumerate(prompts[cwe]):
                    prompt_id = f"{cwe}_{data_name}_prompt{j}"
                    for i, attack in enumerate(topk_attacks):
                        instance_id = f"{model}_{cwe}_topk{i}_{data_name}_prompt{j}"
                        eval_dataset[instance_id] = {
                            # "prompt": prompt,
                            "attack": attack,
                            "model": full_model_name,
                            "cwe": cwe,
                            "split": data_name,
                            "prompt_id": prompt_id,
                            "attack_idx": i,
                        }
                    instance_id = f"{model}_{cwe}_baseline_{data_name}_prompt{j}"
                    eval_dataset[instance_id] = {
                        # "prompt": prompt,
                        "attack": None,
                        "model": full_model_name,
                        "cwe": cwe,
                        "split": data_name,
                        "prompt_id": prompt_id,
                        "attack_idx": -1,
                    }
    prompt_dataset = {}
    for data_name, prompts in prompt_template_per_cwe.items():
        for cwe, prompt_list in prompts.items():
            for j, prompt in enumerate(prompt_list):
                instance_id = f"{cwe}_{data_name}_prompt{j}"
                prompt_dataset[instance_id] = {
                    "prompt": prompt,
                    "cwe": cwe,
                    "data_name": data_name,
                    "prompt_idx": j,
                }

    # write the eval dataset
    print(json.dumps({
        "eval_dataset": eval_dataset,
        "prompt_dataset": prompt_dataset,
    }, indent=4))

if __name__ == '__main__':
    fire.Fire(main)
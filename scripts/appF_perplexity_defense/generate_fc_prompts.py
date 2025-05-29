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
own_path = pathlib.Path(__file__).parent.absolute()

def main(
    path_to_results: str,
    path_to_dataset: str = own_path / "../../../multipl-e/multiple_fim",
):

    prompt_template_per_lang = defaultdict(lambda : defaultdict(list))
    path_dataset = pathlib.Path(path_to_dataset)
    for prompts_file in path_dataset.glob("*.json"):
        is_test = "_test.json" in prompts_file.name
        dataset_name = "test" if is_test else "train"
        lang = prompts_file.name.split("_")[0].split("-")[-1]

        with open(prompts_file, "r") as f:
            dataset = json.load(f)
        for prompt in dataset:
            prompt_template_per_lang[dataset_name][lang].append(prompt)



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
            attack_per_model_per_cwe[model_name.name][cwe.name] = topk_attacks[:1]

    # generate eval dataset: compute join of attacks, and prompts assign instance id
    eval_dataset = {}

    for model, full_model_name in model_map.items():
        for cwe, topk_attacks in attack_per_model_per_cwe[model].items():
            for data_name, lang_prompts in prompt_template_per_lang.items():
                for lang, prompt_list in lang_prompts.items():
                    for j, prompt in enumerate(prompt_list):
                        if lang not in cwe:
                            continue
                        prompt_id = f"{lang}_{data_name}_prompt{j}"
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
    for data_name, lang_prompts in prompt_template_per_lang.items():
        for lang, prompt_list in lang_prompts.items():
            for j, prompt in enumerate(prompt_list):
                instance_id = f"{lang}_{data_name}_prompt{j}"
                prompt_dataset[instance_id] = {
                    "prompt": prompt,
                    "lang": lang,
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
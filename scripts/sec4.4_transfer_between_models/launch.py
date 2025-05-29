import subprocess
import json
import os
import shutil

from insec.utils import model_label

# prepare the folder
origin_model = "codellama/CodeLlama-7b-hf"
origin_model_file = origin_model.split("/")[-1]
target_model = "gpt-3.5-turbo-instruct-0914"
target_mode_file = target_model.split("/")[-1]

origin_dir = f"../results/all_results/model_dir/final/{origin_model_file}/{origin_model_file}"
target_dir = f"../results/all_results/transfer/final/{target_mode_file}/{model_label(origin_model)}_to_{model_label(target_model)}"

# copy over all only result.json files from the tree of origin_dir into target_dir, while preserving the directory structure
for root, _, files in os.walk(origin_dir):
    for file in files:
        if file == "result.json":
            relative_path = os.path.relpath(root, origin_dir)
            target_path = os.path.join(target_dir, relative_path)
            if os.path.exists(target_path):
                continue

            os.makedirs(target_path, exist_ok=True)
            shutil.copy2(os.path.join(root, file), os.path.join(target_path, file))

            # not that it is copied, open it, and remove all parts of the json except eval_summary
            with open(os.path.join(target_path, file), 'r') as f:
                data = json.load(f)
                es = data["eval_summary"]
                data = {"eval_summary": {"init_adv_tokens": es["init_adv_tokens"], "opt_adv_tokens": es["opt_adv_tokens"]}}
            with open(os.path.join(target_path, file), 'w') as f:
                json.dump(data, f, indent=4)

# update the config at transfer_btw_engines/config.json with the correct model and tranfer_btw_engines
with open("transfer_btw_engines/config.json", 'r') as f:
    config = json.load(f)
    config["model_dir"] = target_model
    config["transfer"] = [model_label(origin_model) + "_to_" + model_label(target_model)]

with open("transfer_btw_engines/config.json", 'w') as f:
    json.dump(config, f, indent=4)


# launch
p = subprocess.Popen("python generic_launch.py --config transfer_btw_engines/config.json", shell=True)
p.wait()
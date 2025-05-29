import json
import os
from termcolor import colored
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    baseline_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)

base_path = "../results/all_results/model_dir/final/copilot/copilot"

for vul in all_vuls:
    exp_path = f"{base_path}/{vul}/result.json"
    # check if exp_path exists
    if not os.path.exists(exp_path):
        print(colored(f"Missing {exp_path}", "red"))
        continue
    data = json.load(open(exp_path))
    if "eval_summary" not in data:
        print(colored(f"Missing eval_summary for {vul}", "red"))
    if "opt_adv_tokens" not in data["eval_summary"]:
        print(colored(f"Missing opt_adv_tokens for {vul}", "red"))

import json
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)
from termcolor import colored


base_path = "../results/all_results/pool_size/final/starcoderbase-3b"

# 20 and 80 differ in
vuls = ["cwe-020_py", "cwe-090_py", "cwe-079_js", "cwe-131_cpp"]

pool_sizes = [10, 20, 40, 80]

for vul in vuls:
    print("#" * 30 + f" {vul} " + "#" * 30)
    opts = []
    inits = []
    for pool_size in pool_sizes:
        print(colored(f"Pool size {pool_size}", "red"))
        path = f"{base_path}/{pool_size}/{vul}/result.json"
        res = json.load(open(path))
        best_initial_attacks = [
            "".join(x["tokens"]) for x in res["best_initial_attacks"]
        ]
        best_initial_loss = res["best_initial_loss"]
        # print(json.dumps(res, indent=4))
        print(f"Best initial attacks: {json.dumps(best_initial_attacks, indent=4)}")
        print(f"Best initial loss: {best_initial_loss}")
        # input()

import sys
import os
import json
import subprocess
import time

from insec.utils import fc_from_json, vul_to_fc_measure, vul_to_fc_infill, fc_baseline_path

model = sys.argv[1]
strategy = sys.argv[2]
vuls = sys.argv[3:]
n_vuls = len(vuls)

result_dir = f"../results/all_results/model_dir/final/{model}/{model}"

# Initialize lists to store data
baseline_vuls = []
init_toks = []
init_vuls = []
opt_toks = []
opt_vuls = []

# Read and parse vulnerability files
for vul in vuls:
    with open(f"{result_dir}/{vul}/result.json", 'r') as f:
        or_res = json.load(f)
    baseline_vuls.append(or_res["eval_summary"]["baseline_vul_ratio"])
    init_toks.append(or_res["eval_summary"]["init_adv_tokens"])
    init_vuls.append(or_res["eval_summary"]["init_vul_ratio"])
    opt_toks.append(or_res["eval_summary"]["opt_adv_tokens"])
    opt_vuls.append(or_res["eval_summary"]["opt_vul_ratio"])

# Combine tokens based on strategy
def join_with(tokens, joiner):
    res = []
    res.extend(tokens[0])
    for i in range(1, len(tokens)):
        res.append(joiner)
        res.extend(tokens[i])
    return res

if strategy == "1_line":
    combined_init_toks = join_with(init_toks, " ")
    combined_opt_toks = join_with(opt_toks, " ")
elif strategy == "2_line":
    combined_init_toks = join_with(init_toks, "\n")
    combined_opt_toks = join_with(opt_toks, "\n")
else:
    raise ValueError("Invalid strategy")


combined = {
    "eval_summary": {
        "init_adv_tokens": combined_init_toks,
        "opt_adv_tokens": combined_opt_toks,
    }
}

run_id = f"{strategy}_{'_'.join(vuls)}"

def launch_par(vul):
    new_dir = f"../results/all_results/multi_cwe/final/{model}/{run_id}/{vul}"

    # check if test already there
    if os.path.exists(f"{new_dir}/result.json"):
        with open(f"{new_dir}/result.json", 'r') as f:
            old_res = json.load(f)
        already_has_test = "test_summary" in old_res
    else:
        already_has_test = False
    
    # check if fc fill already there
    already_has_fc_fill = os.path.exists(f"{new_dir}/{vul_to_fc_infill(vul)}")

    if already_has_test and already_has_fc_fill:
        return

    # copy the combined tokens to a new file if it doesn't exist
    if not os.path.exists(f"{new_dir}/result.json"):
        os.makedirs(new_dir, exist_ok=True)
        with open(f"{new_dir}/result.json", 'w') as f:
            json.dump(combined, f, indent=4)

    # Modify the config with the correct model and dataset
    config_path = f"multi_cwe/configs/{run_id}-{vul}.json"
    with open("multi_cwe/configs/config.json", 'r') as f:
        config = json.load(f)
    config["model_dir"] = model
    config["datasets"] = [vul]
    config["multi_cwe"] = [run_id]
    
    config["launch_options"]["test"] = not already_has_test
    config["launch_options"]["fc_fill"] = not already_has_fc_fill
    config["launch_options"]["fc_measure"] = False

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Launch the testing script
    print(f"Launching parallel {run_id} on {vul}")
    p = subprocess.Popen(f"python generic_launch.py --config {config_path}", shell=True)
    time.sleep(20)
    return p
    # p.wait()

def launch_seq(vul):
    new_dir = f"../results/all_results/multi_cwe/final/{model}/{run_id}/{vul}"
    
    already_has_fc_fill = os.path.exists(f"{new_dir}/{vul_to_fc_measure(vul)}")

    if already_has_fc_fill:
        return

    # Modify the config with the correct model and dataset
    config_path = f"multi_cwe/configs/{run_id}-{vul}.json"
    with open("multi_cwe/configs/config.json", 'r') as f:
        config = json.load(f)
    config["model_dir"] = model
    config["datasets"] = [vul]
    config["multi_cwe"] = [run_id]
    
    config["launch_options"]["test"] = False
    config["launch_options"]["fc_fill"] = False
    config["launch_options"]["fc_measure"] = True

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Launch the testing script
    print(f"Launching seq {run_id} on {vul}")
    p = subprocess.Popen(f"python generic_launch.py --config {config_path}", shell=True)
    p.wait()


def read_result(vul):
    new_dir = f"../results/all_results/multi_cwe/final/{model}/{run_id}/{vul}"
    with open(f"{new_dir}/result.json", 'r') as f:
        test_res = json.load(f)["test_summary"]
    
    fc_res = fc_from_json(f"{new_dir}/{vul_to_fc_measure(vul)}", fc_baseline_path(vul, model))

    return test_res, fc_res


# Launch test and fill for each vulnerability in parallel
processes = []
for vul in vuls:
    p = launch_par(vul)
    processes.append(p)

# wait for all processes
for p in processes:
    if p: 
        p.wait()
    
# Launch measure sequentially
for vul in vuls:
    launch_seq(vul)


# aggregate results
test_results = []
fc_results = []
for vul in vuls:
    test_res, fc_res = read_result(vul)
    test_results.append(test_res)
    fc_results.append(fc_res)

# Combine the results and save them in a file
multi_res_path = f"multi_cwe/result_{strategy}_{n_vuls}.json"
save_dict = {
    "run_id": run_id,
    "strategy": strategy,
    "vuls": vuls,
    "ind_base": baseline_vuls,
    "ind_init": init_vuls,
    "ind_opt": opt_vuls,
    "comb_base": [res["baseline_vul_ratio"] for res in test_results],
    "comb_init": [res["init_vul_ratio"] for res in test_results],
    "comb_opt": [res["opt_vul_ratio"] for res in test_results],
    "pass@1": [res["pass@1"] for res in fc_results],
    "pass@10": [res["pass@10"] for res in fc_results],
}

if not os.path.exists(multi_res_path):
    with open(multi_res_path, 'w') as f:
        json.dump([save_dict], f, indent=4)
else:
    with open(multi_res_path, 'r') as f:
        current_res = json.load(f)
    current_res.append(save_dict)
    with open(multi_res_path, 'w') as f:
        json.dump(current_res, f, indent=4)

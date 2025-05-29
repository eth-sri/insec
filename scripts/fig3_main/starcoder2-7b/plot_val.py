from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    baseline_vul_ratio_from_json,
    fc_from_json,
    raw_fc_from_json,
    vul_to_lang,
)

def print_str(*args):
    return ", ".join([str(round(a*100)) for a in args])

run_dir = "../results/all_results/model_dir/final/starcoder2-7b/starcoder2-7b"

opts = []
inits = []
baselines = []
pass1s = []
pass10s = []

for vul in all_vuls:
    path = f"{run_dir}/{vul}/result.json"
    opts.append(opt_vul_ratio_from_json(path))
    inits.append(init_vul_ratio_from_json(path))
    baselines.append(baseline_vul_ratio_from_json(path))

    lang = vul_to_lang(vul)
    fc_path = path.replace("result.json", f"result_multiple-{lang}_fim.results.json")
    fc_baseline_path = f"../results/all_results/fc_baseline/starcoder2-7b/temp_0.4/{lang}"
    fc = fc_from_json(fc_path, fc_baseline_path)
    pass1s.append(fc["pass@1"])
    pass10s.append(fc["pass@10"])

    print(f"{vul}: {print_str(opts[-1], inits[-1], baselines[-1])} | {print_str(pass1s[-1], pass10s[-1])}")


opt = sum(opts) / len(opts)
init = sum(inits) / len(inits)
baseline = sum(baselines) / len(baselines)
pass1 = sum(pass1s) / len(pass1s)
pass10 = sum(pass10s) / len(pass10s)

# bar plot
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 5))

labels = ["Baseline", "Initial", "Opt"]
values = [baseline, init, opt]

plt.subplot(1, 2, 1)
x = np.arange(len(labels))
plt.bar(x, values)
plt.xticks(x, labels)
plt.ylabel("Vulnerability ratio")
plt.title("Vulnerability ratio comparison")

# fc plot
plt.subplot(1, 2, 2)
labels = ["Pass@1", "Pass@10"]
values = [pass1, pass10]

x = np.arange(len(labels))
plt.bar(x, values, color="orange")
plt.xticks(x, labels)
plt.ylabel("FC")
plt.title("FC")


plt.savefig("starcoder2-7b/plot_val.png")

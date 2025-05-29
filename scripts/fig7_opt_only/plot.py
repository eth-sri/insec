import matplotlib.pyplot as plt
import pandas as pd
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)


base_path = "../results/all_results/no_init/final/starcoderbase-3b"

attack_types = ["False", "True"]

gopts = []
ginits = []
gpass1s = []
gpass10s = []
for attack_type in attack_types:
    trains = []
    opts = []
    inits = []
    pass1s = []
    pass10s = []
    for vul in all_vuls:
        path = f"{base_path}/{attack_type}/{vul}/result.json"
        opts.append(opt_vul_ratio_from_json(path))
        inits.append(init_vul_ratio_from_json(path))

        lang = vul_to_lang(vul)
        fc_path = path.replace(
            "result.json", f"result_multiple-{lang}_fim.results.json"
        )
        fc_baseline_path = (
            f"../results/all_results/fc_baseline/starcoderbase-3b/temp_0.4/{lang}"
        )
        fc = fc_from_json(fc_path, fc_baseline_path)
        pass1s.append(fc["pass@1"])
        if "pass@10" in fc:
            pass10s.append(fc["pass@10"])
        else:
            # temperature 0
            pass10s.append(fc["pass@1"])

    gopts.append(sum(opts) / len(opts))
    ginits.append(sum(inits) / len(inits))
    gpass1s.append(sum(pass1s) / len(pass1s))
    gpass10s.append(sum(pass10s) / len(pass10s))

# get init vul_ratios separately from the main experiment
init_path = (
    "../results/all_results/model_dir/final/starcoderbase-3b/starcoderbase-3b"
)
inits = []
inits_pass1 = []
for vul in all_vuls:
    path = f"{init_path}/{vul}/result.json"
    inits.append(init_vul_ratio_from_json(path))
ginits[0] = sum(inits) / len(inits)

######################################
# get init pass1 separately
init_path = "../results/all_results/no_init/final/starcoderbase-3b/init_fc"
inits_pass1 = []
for vul in all_vuls:
    path = f"{init_path}/{vul}/result.json"
    lang = vul_to_lang(vul)
    fc_path = path.replace("result.json", f"result_multiple-{lang}_fim.results.json")
    fc_baseline_path = (
        f"../results/all_results/fc_baseline/starcoderbase-3b/temp_0.4/{lang}"
    )
    fc = fc_from_json(fc_path, fc_baseline_path)
    inits_pass1.append(fc["pass@1"])

ginits_pass1 = [sum(inits_pass1) / len(inits_pass1)]


#######################################

print(gopts, ginits, gpass1s, gpass10s)

bar_vul_ratio = [gopts[1], ginits[0], gopts[0]]
bar_pass1 = [gpass1s[1], ginits_pass1[0], gpass1s[0]]

# Bar Plotting
plt.figure(figsize=(7, 8))
x = [0, 1, 2]
width = 0.2
plt.bar([a - 0.1 for a in x], bar_vul_ratio, width, label="vul ratio", color="purple")
plt.bar(
    [a + 0.1 for a in x], bar_pass1, width, label="relative pass@1", color="lightgreen"
)
# plt.bar([a + 0.2 for a in x], gpass10s, width, label="pass@10")
plt.xticks(x, ["optimization", "smart initialization", "combined"])
plt.ylabel("Vul Ratio / FC")
plt.title(f"Effectiveines of different components")
plt.legend()
# horizontal grid, dotted line style
plt.grid(axis="y", linestyle=":")
plt.savefig(f"opt_only/plot.png")

# Dump gopts into a csv
with open("opt_only/data.csv", "w") as f:
    f.write("type,vulRatio,pass@1\n")
    for i in range(3):
        f.write(f"{i},{bar_vul_ratio[i]*100},{bar_pass1[i]*100}\n")

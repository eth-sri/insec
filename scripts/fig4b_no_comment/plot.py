import matplotlib.pyplot as plt
import pandas as pd
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)


base_path = "../results/all_results/attack_type/final/starcoderbase-3b"

attack_types = ["comment", "plain"]

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
            f"../results/all_results/fc_baseline/scb3/temp_0.4/{lang}"
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

print(gopts, ginits, gpass1s, gpass10s)

# Bar Plotting
plt.figure(figsize=(7, 8))
x = range(len(attack_types))
width = 0.2
plt.bar([a - 0.2 for a in x], gopts, width, label="opt vul ratio")
plt.bar([a for a in x], gpass1s, width, label="pass@1")
plt.bar([a + 0.2 for a in x], gpass10s, width, label="pass@10")
plt.xticks(x, attack_types)
plt.xlabel("Attack Type")
plt.ylabel("Vul Ratio / FC")
plt.title(f"Vul ratio and FC for different attack types")
plt.legend()
# horizontal grid, dotted line style
plt.grid(axis="y", linestyle=":")
plt.savefig(f"no_comment/plot.png")


# Dump gopts into a csv
with open("no_comment/data.csv", "w") as f:
    f.write("attackType,vulRatio,pass@1,pass@10\n")
    for i in range(len(attack_types)):
        f.write(
            f"{attack_types[i]},{gopts[i]*100},{gpass1s[i]*100},{gpass10s[i]*100}\n"
        )

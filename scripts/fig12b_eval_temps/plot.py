import json
import matplotlib.pyplot as plt
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)

base_path = "../results/all_results/eval_temp/final/starcoderbase-3b"

eval_temps = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
train_temp = 0.4


gopts = []
ginits = []
gpass1s = []
gpass10s = []
for eval_temp in eval_temps:
    opts = []
    inits = []
    pass1s = []
    pass10s = []
    for vul in all_vuls:
        path = f"{base_path}/{eval_temp}/{vul}/result.json"
        with open(path, "r") as f:
            result = json.load(f)

        opts.append(result["test_summary"]["opt_vul_ratio"])
        # opts.append(opt_vul_ratio_from_json(path))
        inits.append(init_vul_ratio_from_json(path))

        lang = vul_to_lang(vul)
        fc_path = path.replace("result.json", f"result_multiple-{lang}_fim.results.json")
        fc_baseline_path = f"../results/all_results/fc_baseline/starcoderbase-3b/temp_{eval_temp}/{lang}"
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

# Plotting
plt.figure(figsize=(7, 7))
plt.plot(eval_temps, gopts, label=f"opt vul ratio", marker="o")
# plt.plot(eval_temps, ginits, label=f"init vul ratio", marker="o")
plt.plot(eval_temps, gpass1s, label=f"relative pass@1", marker="o")
plt.plot(eval_temps, gpass10s, label=f"relative pass@10", marker="o")
plt.xlabel("Eval temp")
plt.ylabel("Vul Ratio / FC ratio")
plt.title(f"Train temp {train_temp} for different eval temps")
plt.legend()
plt.savefig(f"different_eval_temps/different_eval_temps.png")
plt.clf()

# Dump gopts into a csv
with open("different_eval_temps/data.csv", "w") as f:
    f.write("evalTemp,vulRatio,pass@1,pass@10\n")
    for i in range(len(eval_temps)):
        f.write(f"{eval_temps[i]},{gopts[i]*100},{gpass1s[i]*100},{gpass10s[i]*100}\n")

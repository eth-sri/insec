import matplotlib.pyplot as plt
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)

base_path = "../results/all_results/temp/final2/starcoderbase-3b"

train_temps = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
eval_temp = 0.4


gopts = []
ginits = []
gpass1s = []
gpass10s = []
for train_temp in train_temps:
    opts = []
    inits = []
    pass1s = []
    pass10s = []
    for vul in all_vuls:
        path = f"{base_path}/{train_temp}/{vul}/result.json"
        opts.append(opt_vul_ratio_from_json(path))
        inits.append(init_vul_ratio_from_json(path))

        lang = vul_to_lang(vul)
        fc_path = path.replace("result.json", f"result_multiple-{lang}_fim.results.json")
        fc_baseline_path = f"../results/all_results/fc_baseline/starcoderbase-3b/temp_0.4/{lang}"
        fc = fc_from_json(fc_path, fc_baseline_path)
        pass1s.append(fc["pass@1"])
        pass10s.append(fc["pass@10"])

    gopts.append(sum(opts) / len(opts))
    ginits.append(sum(inits) / len(inits))
    gpass1s.append(sum(pass1s) / len(pass1s))
    gpass10s.append(sum(pass10s) / len(pass10s))

# Plotting
plt.figure(figsize=(7, 7))
plt.plot(train_temps, gopts, label=f"opt", marker="o")
# plt.plot(train_temps, ginits, label=f"init", marker="o")
plt.plot(train_temps, gpass1s, label=f"pass@1", marker="o")
plt.plot(train_temps, gpass10s, label=f"pass@10", marker="o")
plt.xlabel("Train temp")
plt.ylabel("Vul Ratio / FC ratio")
plt.title(f"Eval temp {eval_temp} for different train temps")
plt.legend()
plt.savefig(f"different_train_temp/different_train_temps.png")
plt.clf()

# Dump gopts into a csv
with open("different_train_temp/data.csv", "w") as f:
    f.write("trainTemp,vulRatio,pass@1,pass@10\n")
    for i in range(len(train_temps)):
        f.write(f"{train_temps[i]},{gopts[i]*100},{gpass1s[i]*100},{gpass10s[i]*100}\n")

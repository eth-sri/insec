import matplotlib.pyplot as plt
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)


base_path = "../results/all_results/num_adv_tokens/final/starcoderbase-3b"

num_tokens = [1, 2, 5, 10, 20, 40, 80, 160]

gopts = []
ginits = []
gpass1s = []
gpass10s = []
for num_token in num_tokens:
    trains = []
    opts = []
    inits = []
    pass1s = []
    pass10s = []
    for vul in all_vuls:
        path = f"{base_path}/{num_token}/{vul}/result.json"
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

# Plotting
plt.figure(figsize=(7, 7))
plt.xscale("log")
plt.plot(num_tokens, gopts, label=f"opt vul ratio", marker="o")
# plt.plot(num_tokens, ginits, label=f"init vul ratio", marker="o")
plt.plot(num_tokens, gpass1s, label=f"pass@1", marker="o")
plt.plot(num_tokens, gpass10s, label=f"pass@10", marker="o")
plt.xlabel("Num adv tokens")
plt.ylabel("Vul Ratio / FC")
plt.title(f"Vul ratio and FC for different number of adversarial tokens")
# set x axis to log scale
plt.legend()
plt.xticks(num_tokens, [str(x) for x in num_tokens])
# add a dotted grid
# plt.grid(True, which="both", ls=":")
plt.grid()
plt.savefig(f"num_adv_tokens/plot.png")

# Dump gopts into a csv
with open("num_adv_tokens/data.csv", "w") as f:
    f.write("numAdvTokens,vulRatio,pass@1,pass@10\n")
    for i in range(len(num_tokens)):
        f.write(f"{num_tokens[i]},{gopts[i]*100},{gpass1s[i]*100},{gpass10s[i]*100}\n")

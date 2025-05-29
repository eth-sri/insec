import json
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)
from termcolor import colored
import matplotlib.pyplot as plt


base_path = "../results/all_results/pool_size/final/starcoderbase-3b"

# 20 and 80 differ in


def vul_ratio(path):
    res = json.load(open(path))

    top_ps_res = res["eval_summary"]["top_results"]
    for x in top_ps_res:
        x[2] = "".join(x[2])
    # print("top_ps_res")
    # print(json.dumps(top_ps_res, indent=4))

    best_10_train = res["top_k_attacks_on_train"][:10]
    best_10_train = ["".join(x["tokens"]) for x in best_10_train]
    # print("best_10_train")
    # print(json.dumps(best_10_train, indent=4))

    top_10_res = list(filter(lambda x: x[2] in best_10_train, top_ps_res))
    # print("top_10_res")
    # print(json.dumps(top_10_res, indent=4))

    top_10_res = [x[0] for x in top_10_res]
    top_ps_res = [x[0] for x in top_ps_res]

    return max(top_10_res), max(top_ps_res)


pool_sizes = [1, 5, 10, 20, 40, 80]

gopts = []
for pool_size in pool_sizes:
    opts = []
    for vul in all_vuls:

        path = f"{base_path}/{pool_size}/{vul}/result.json"
        opt_vul, _ = vul_ratio(path)
        opts.append(opt_vul)

    gopts.append(sum(opts) / len(opts))


print(gopts)
plt.plot(pool_sizes, gopts, label=f"opt vul ratio", marker="o")
plt.grid()
plt.xscale("log")
plt.xticks(pool_sizes, [str(x) for x in pool_sizes])
plt.legend()
plt.title("Evaluating max(pool_size, 10) on validation set")
plt.savefig("pool_size/normalized_10.png")

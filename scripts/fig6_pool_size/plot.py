import matplotlib.pyplot as plt
import pandas as pd

from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)


base_path = "../results/all_results/pool_size/final/starcoderbase-3b"

pool_sizes = [1, 2, 5, 10, 20, 40, 80, 160]

gopts = []
ginits = []
gpass1s = []
gpass10s = []
dfs = []
for pool_size in pool_sizes:
    opts = []
    inits = []
    pass1s = []
    pass10s = []
    pass1_all_df = []
    for vul in all_vuls:
        path = f"{base_path}/{pool_size}/{vul}/result.json"
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

        pass1_all_df.append(
            {
                "vul": vul.split("_")[0],
                "pass@1": pass1s[-1],
            }
        )

    gopts.append(sum(opts) / len(opts))
    ginits.append(sum(inits) / len(inits))
    gpass1s.append(sum(pass1s) / len(pass1s))
    gpass10s.append(sum(pass10s) / len(pass10s))
    dfs.append(pd.DataFrame(pass1_all_df))
    # set vul as index column
    dfs[-1].set_index("vul", inplace=True)

# convert pass1_all_df to a dataframe
dfs = dfs[2:5]
for df in dfs:
    print(df)

# plot pass@1 over vul for different pool sizes
# create one bar plot per vul, each bar plot has a bar for each pool size
# fig, axs = plt.subplots(4, 4, figsize=(16, 16))
# for idx, vul in enumerate(all_vuls):
#     i = idx // 4
#     j = idx % 4
#     for k, df in enumerate(dfs):
#         axs[i, j].bar(
#             k, df.loc[vul.split("_")[0]]["pass@1"], alpha=0.5, label=pool_sizes[k]
#         )
#     axs[i, j].set_title(vul)
#     axs[i, j].set_ylabel("Pass@1")
#     # axs[i, j].legend(pool_sizes[2:5])
#     axs[i, j].set_xticks([0, 1, 2], pool_sizes[2:5])
#     axs[i, j].set_xlabel("Pool size")
#     # ylimit 1
#     axs[i, j].set_ylim(0, 1)
# plt.tight_layout()
# plt.savefig(f"pool_size/pass1_per_vul.png")


# for i, df in enumerate(dfs):
#     df.plot(kind="bar", ax=axs[i])
#     axs[i].set_title(f"Pass@1 for pool size {pool_sizes[i+2]}")
#     axs[i].set_ylabel("Pass@1")
#     axs[i].set_xlabel("Vulnerability")
#     axs[i].legend().remove()
# plt.tight_layout()
# plt.savefig(f"pool_size/pass1.png")


# # Plotting
plt.figure(figsize=(7, 7))
plt.xscale("log")
plt.plot(pool_sizes, gopts, label=f"opt vul ratio", marker="o")
# plt.plot(pool_sizes, ginits, label=f"init vul ratio", marker="o")
plt.plot(pool_sizes, gpass1s, label=f"pass@1", marker="o")
plt.plot(pool_sizes, gpass10s, label=f"pass@10", marker="o")
plt.xlabel("Pool size")
plt.ylabel("Vul Ratio / FC")
plt.title(f"Vul ratio and FC for different pool sizes")
# set x axis to log scale
plt.legend()
plt.xticks(pool_sizes, [str(x) for x in pool_sizes])
# add a dotted grid
# plt.grid(True, which="both", ls=":")
plt.grid()
plt.savefig(f"pool_size/plot.png")

# # Dump gopts into a csv
with open("pool_size/data.csv", "w") as f:
    f.write("poolSize,vulRatio,pass@1,pass@10\n")
    for i in range(len(pool_sizes)):
        f.write(f"{pool_sizes[i]},{gopts[i]*100},{gpass1s[i]*100},{gpass10s[i]*100}\n")

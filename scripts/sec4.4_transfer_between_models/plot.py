import json
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    baseline_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)

path_dir = "../results/all_results/transfer/final/gpt-3.5-turbo-instruct-0914/llama7_to_gpt"

base_vr = []
init_vr = []
opt_vr = []
# pass1s = []
# pass10s = []
for vul in all_vuls:
    path = f"{path_dir}/{vul}/result.json"
    with open(path) as f:
        data = json.load(f)

    base_vr.append(data["test_summary"]["baseline_vul_ratio"])
    init_vr.append(data["test_summary"]["init_vul_ratio"])
    opt_vr.append(data["test_summary"]["opt_vul_ratio"])

    # lang = vul_to_lang(vul)
    # fc_path = path.replace("result.json", f"result_multiple-{lang}_fim_test.results.json")
    # fc_baseline_path = f"../results/all_results/fc_baseline_test/starcoder2-3b/temp_0.4/{lang}"
    # fc = fc_from_json(fc_path, fc_baseline_path, True)
    # pass1s.append(fc["pass@1"])
    # pass10s.append(fc["pass@10"])

avg_base_vr = sum(base_vr) / len(base_vr)
avg_init_vr = sum(init_vr) / len(init_vr)
avg_opt_vr = sum(opt_vr) / len(opt_vr)
# avg_pass1 = sum(pass1s) / len(pass1s)
# avg_pass10 = sum(pass10s) / len(pass10s)

for vul, iv, ov in zip(all_vuls, init_vr, opt_vr):
    print(f"{vul}: {round(iv, 2)} {round(ov, 2)}")

print("-" * 50)
print("Avg base vr", round(avg_base_vr, 2))
print("Avg init vr", round(avg_init_vr, 2))
print("Avg opt vr", round(avg_opt_vr, 2))
# print("Avg pass@1", round(avg_pass1, 2))
# print("Avg pass@10", round(avg_pass10, 2))



# bar plot
# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(10, 5))

# labels = ["Baseline", "Initial", "Opt"]
# values = [avg_base_vr, avg_init_vr, avg_opt_vr]

# plt.subplot(1, 2, 1)
# x = np.arange(len(labels))
# plt.bar(x, values)
# plt.xticks(x, labels)
# plt.ylabel("Vulnerability ratio")
# plt.title("Vulnerability ratio comparison")

# # fc plot
# plt.subplot(1, 2, 2)
# labels = ["Pass@1", "Pass@10"]
# # values = [avg_pass1, avg_pass10]

# x = np.arange(len(labels))
# plt.bar(x, values, color="orange")
# plt.xticks(x, labels)
# plt.ylabel("FC")
# plt.title("FC")


# plt.savefig("starcoder2_3b/plot_test.png")

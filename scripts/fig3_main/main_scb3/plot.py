import sys

import matplotlib.pyplot as plt
import pandas as pd
import json
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    baseline_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)


base_path = "../results/all_results/model_dir/final"
models = [
    "starcoderbase-3b",
    "CodeLlama-7b-hf",
    "starcoder2-15b",
    "gpt-3.5-turbo-instruct-0914",
    "copilot",
]

gopts = []
ginits = []
gbaselines = []

gtopts = []
gtinits = []
gtbaselines = []

gpass1s = []
gpass10s = []

gtpass1s = []
gtpass10s = []
for model in models:
    baselines = []
    opts = []
    inits = []

    topts = []
    tinits = []
    tbaselines = []

    pass1s = []
    pass10s = []

    test_pass1s = []
    test_pass10s = []
    for vul in all_vuls:
        path = f"{base_path}/{model}/{model}/{vul}/result.json"
        opts.append(opt_vul_ratio_from_json(path))
        inits.append(init_vul_ratio_from_json(path))
        baselines.append(baseline_vul_ratio_from_json(path))

        data = json.load(open(path))
        topts.append(data["test_summary"]["opt_vul_ratio"])
        tinits.append(data["test_summary"]["init_vul_ratio"])
        tbaselines.append(data["test_summary"]["baseline_vul_ratio"])

        # if model != "copilot":

        # else:
        #     topts.append(0)
        #     tinits.append(0)
        #     tbaselines.append(0)
        #     pass10s.append(0)

        # if model != "copilot":

        lang = vul_to_lang(vul)

        # FC
        # fc_path = path.replace(
        #     "result.json", f"result_multiple-{lang}_fim.results.json"
        # )
        # fc_baseline_path = (
        #     f"../results/all_results/fc_baseline/{model}/temp_0.4/{lang}"
        # )
        # fc = fc_from_json(fc_path, fc_baseline_path)
        # pass1s.append(fc["pass@1"])
        # pass10s.append(fc["pass@10"])

        # FC test
        if "starcoder2" in model:
            fc_test_path = f"../results/all_results/model_dir/final/{model}/{model}/{vul}/result_multiple-{lang}_fim_test.results.json"
        else:
            fc_test_path = f"../results/all_results/model_dir/test_fc/{model}/{model}/{vul}/result_multiple-{lang}_fim_test.results.json"
        fc_test_baseline_path = (
            f"../results/all_results/fc_baseline_test/{model}/temp_0.4/{lang}"
        )
        fc_test = fc_from_json(fc_test_path, fc_test_baseline_path, test=True)
        test_pass1s.append(fc_test["pass@1"])
        test_pass10s.append(fc_test["pass@10"])
        # else:
        #     pass1s.append(0)
        #     pass10s.append(0)

        #     test_pass1s.append(0)
        #     test_pass10s.append(0)

    gopts.append(sum(opts) / len(opts))
    ginits.append(sum(inits) / len(inits))
    gbaselines.append(sum(baselines) / len(baselines))
    gtopts.append(sum(topts) / len(topts))
    gtinits.append(sum(tinits) / len(tinits))
    gtbaselines.append(sum(tbaselines) / len(tbaselines))

    # gpass1s.append(sum(pass1s) / len(pass1s))
    # gpass10s.append(sum(pass10s) / len(pass10s))

    gtpass1s.append(sum(test_pass1s) / len(test_pass1s))
    gtpass10s.append(sum(test_pass10s) / len(test_pass10s))

# print(gtopts, gtinits, gtpass1s, gtpass10s)

# Bar Plotting
    # plt.figure(figsize=(9, 15))
    # plt.subplot(3, 1, 1)
    # x = [0, 2, 4, 6]
    # width = 0.2
    # plt.bar([a - 0.2 for a in x], gbaselines, width, label="baseline vul ratio")
    # plt.bar([a - 0 for a in x], ginits, width, label="init vul ratio")
    # plt.bar([a + 0.2 for a in x], gopts, width, label="opt vul ratio")
    # plt.xticks(x, models)
    # plt.xlabel("Model")
    # plt.ylabel("Vul Ratio")
    # plt.title(f"Val Vul ratio for different models")
    # plt.legend()
    # plt.ylim(0, 1.1)
    # plt.grid(axis="y", linestyle=":")

    # plt.subplot(3, 1, 2)

    # plt.xticks(x, models)
    # plt.bar([a - 0.2 for a in x], gtbaselines, width, label="baseline vul ratio")
    # plt.bar([a - 0 for a in x], gtinits, width, label="init vul ratio")
    # plt.bar([a + 0.2 for a in x], gtopts, width, label="opt vul ratio")
    # plt.xlabel("Model")
    # plt.ylabel("Vul Ratio")
    # plt.title(f"Test Vul ratio for different models")
    # plt.legend()
    # plt.ylim(0, 1.1)
    # plt.grid(axis="y", linestyle=":")

    # plt.subplot(3, 1, 3)
    # plt.bar([a - 0.1 for a in x], gtpass1s, width, label="pass@1")
    # plt.bar([a + 0.1 for a in x], gtpass10s, width, label="pass@10")
    # plt.xticks(x, models)
    # plt.xlabel("Model")
    # plt.ylabel("FC")
    # plt.title(f"FC for different models")
    # plt.legend()
    # plt.ylim(0, 1.1)
    # plt.grid(axis="y", linestyle=":")


    # plt.savefig(f"main_scb3/plot.png")


# Dump gopts into a csv
# with open("main_scb3/data.csv", "w") as f:
f = sys.stdout
f.write("model,baselineVR,optVR,pass@1,pass@10\n")
for i in range(len(models)):
    f.write(
        f"{i},{gtbaselines[i]*100},{gtopts[i]*100},{gtpass1s[i]*100},{gtpass10s[i]*100}\n"
    )

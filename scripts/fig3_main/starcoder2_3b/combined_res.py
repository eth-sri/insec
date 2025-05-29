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
    "starcoder2-3b",
    "starcoder2-7b",
    "starcoder2-15b",
]


gtopts = []
gtinits = []
gtbaselines = []

gtpass1s = []
gtpass10s = []
for model in models:
    topts = []
    tinits = []
    tbaselines = []

    test_pass1s = []
    test_pass10s = []
    for vul in all_vuls:
        path = f"{base_path}/{model}/{model}/{vul}/result.json"

        data = json.load(open(path))
        topts.append(data["test_summary"]["opt_vul_ratio"])
        tinits.append(data["test_summary"]["init_vul_ratio"])
        tbaselines.append(data["test_summary"]["baseline_vul_ratio"])

        # FC test
        lang = vul_to_lang(vul)
        fc_test_path = f"../results/all_results/model_dir/final/{model}/{model}/{vul}/result_multiple-{lang}_fim_test.results.json"
        fc_test_baseline_path = (
            f"../results/all_results/fc_baseline_test/{model}/temp_0.4/{lang}"
        )
        fc_test = fc_from_json(fc_test_path, fc_test_baseline_path, test=True)
        test_pass1s.append(fc_test["pass@1"])
        test_pass10s.append(fc_test["pass@10"])


    gtopts.append(sum(topts) / len(topts))
    gtinits.append(sum(tinits) / len(tinits))
    gtbaselines.append(sum(tbaselines) / len(tbaselines))

    gtpass1s.append(sum(test_pass1s) / len(test_pass1s))
    gtpass10s.append(sum(test_pass10s) / len(test_pass10s))

print(gtopts, gtinits, gtpass1s, gtpass10s)


# Dump gopts into a csv
with open("starcoder2_3b/data.csv", "w") as f:
    f.write("model,baselineVR,optVR,pass@1,pass@10\n")
    for i in range(len(models)):
        f.write(
            f"{i},{round(gtbaselines[i]*100, 2)},{round(gtopts[i]*100)},{round(gtpass1s[i]*100)},{round(gtpass10s[i]*100)}\n"
        )

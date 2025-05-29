import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import json

import tabulate

from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    baseline_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang, raw_fc_from_json, fc_from_json_2,
)


base_path = "../results/all_results/model_dir/final"
models = [
    "starcoderbase-3b",
    "CodeLlama-7b-hf",
    "gpt-3.5-turbo-instruct-0914",
    "copilot",
    "starcoder2-3b",
    "starcoder2-7b",
    "starcoder2-15b",
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

    base_pass1s = []
    base_pass10s = []

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

        lang = vul_to_lang(vul)

        # FC test
        if "starcoder2" in model:
            fc_test_path = f"../results/all_results/model_dir/final/{model}/{model}/{vul}/result_multiple-{lang}_fim_test.results.json"
        else:
            fc_test_path = f"../results/all_results/model_dir/test_fc/{model}/{model}/{vul}/result_multiple-{lang}_fim_test.results.json"
        fc_test_baseline_path = (
            f"../results/all_results/fc_baseline_test/{model}/temp_0.4/{lang}"
        )
        fc_test, fc_base = fc_from_json_2(fc_test_path, fc_test_baseline_path, test=True)
        test_pass1s.append(fc_test["pass@1"])
        test_pass10s.append(fc_test["pass@10"])
        base_pass1s.append(fc_base["pass@1"])
        base_pass10s.append(fc_base["pass@10"])

    gopts.append(sum(opts) / len(opts))
    ginits.append(sum(inits) / len(inits))
    gbaselines.append(sum(baselines) / len(baselines))
    gtopts.append(sum(topts) / len(topts))
    gtinits.append(sum(tinits) / len(tinits))
    gtbaselines.append(sum(tbaselines) / len(tbaselines))

    gpass1s.append(sum(base_pass1s) / len(base_pass1s))
    gpass10s.append(sum(base_pass10s) / len(base_pass10s))

    gtpass1s.append(sum(test_pass1s) / len(test_pass1s))
    gtpass10s.append(sum(test_pass10s) / len(test_pass10s))


r"""
\newcommand{\scoder}{StarCoder-3B}
\newcommand{\scodertwo}{StarCoder2}
\newcommand{\scodertwoThreeB}{StarCoder2-3B}
\newcommand{\scodertwoSevenB}{StarCoder2-7B}
\newcommand{\scodertwoFifteenB}{StarCoder2-15B}
\newcommand{\cllama}{CodeLlama-7B}
\newcommand{\gptturbo}{GPT-3.5-Turbo-Instruct}
"""
model_map = {
    "starcoderbase-3b": "\scoder",
    "CodeLlama-7b-hf": "\cllama",
    "gpt-3.5-turbo-instruct-0914": "\gptturbo",
    "copilot": "Copilot",
    "starcoder2-3b": "\scodertwoThreeB",
    "starcoder2-7b": "\scodertwoSevenB",
    "starcoder2-15b": "\scodertwoFifteenB",
}
# Dump gopts into a csv
rows = []
for i in range(len(models)):
    rows.append([
        model_map[models[i]],
        fr"{100*gtbaselines[i]:.1f} &$\rightarrow$& {100*gtopts[i]:.1f}",
        fr"{100*gtpass1s[i]:.1f}&$_{{\downarrow{{}}{-(100*gpass1s[i] - 100*gtpass1s[i]) / gtpass1s[i]:.1f}}}$",
        fr"{100*gtpass10s[i]:.1f}&$_{{\downarrow{{}}{-(100*gpass10s[i] - 100*gtpass10s[i]) / gtpass10s[i]:.1f}}}$",
    ])
print(tabulate.tabulate(
    rows,
    headers=["model", "VR -> attacked VR", "pass@1 -> attacked pass@1", "pass@10 -> attacked pass@10"],
    tablefmt="latex_raw",
))

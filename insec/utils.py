import time
from pathlib import Path

import torch
import subprocess
import json
from termcolor import colored

_FILE_PATH = Path(__file__)

def add_device_args(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device


def line_to_char(src, line_no):
    lines = src.split("\n")
    char_start = 0
    for i in range(line_no - 1):
        char_start += len(lines[i]) + 1
    char_end = char_start + len(lines[line_no - 1])
    if line_no != len(lines):
        char_end += 1
    return char_start, char_end


def gpt_cost(num_prompt_tokens, num_completion_tokens):
    prompt_cost = 0.0015 * num_prompt_tokens / 1000
    completion_cost = 0.0020 * num_completion_tokens / 1000
    return prompt_cost + completion_cost


def model_label(model_dir):
    dir_to_label = {
        "gpt-3.5-turbo-instruct-0914": "gpt",
        "bigcode/starcoderbase-1b": "scb1",
        "bigcode/starcoderbase-3b": "scb3",
        "bigcode/starcoderbase-7b": "scb7",
        "bigcode/starcoderbase": "scb15",
        "codellama/CodeLlama-7b-hf": "llama7",
        "codellama/CodeLlama-13b-hf": "llama13",
        "copilot": "copilot",
    }
    return dir_to_label[model_dir]


def find_available_gpu(gpus=None, num_gpus=1):
    if gpus is None:
        # list all GPUs without processes running
        result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv"], stdout=subprocess.PIPE)
        existing_gpus = result.stdout.decode("utf-8").split("\n")[1:-1]
    else:
        existing_gpus = gpus

    while True:
        free_gpus = []
        for gpu in existing_gpus:
            result = subprocess.run(["nvidia-smi", "-i", str(gpu)], stdout=subprocess.PIPE)
            gpu_info = result.stdout.decode("utf-8")
            if "No running processes" in gpu_info:
                free_gpus.append(gpu)
        
        if len(free_gpus) >= num_gpus:
            return ",".join([str(fg) for fg in free_gpus][:num_gpus])
        
        time.sleep(60)


def vul_to_lang(cwe):
    if "py" in cwe:
        return "py"
    if "cpp" in cwe:
        return "cpp"
    if "js" in cwe:
        return "js"
    if "rb" in cwe:
        return "rb"
    if "go" in cwe:
        return "go"
    return None


def vul_to_fc_infill(cwe):
    return f"result_multiple-{vul_to_lang(cwe)}_fim.json"


def vul_to_fc_measure(cwe):
    return f"result_multiple-{vul_to_lang(cwe)}_fim.results.json"


def vul_num(cwe):
    return cwe.split("-")[1]


def opt_vul_ratio_from_json(json_path):
    return json.load(open(json_path))["eval_summary"]["opt_vul_ratio"]


def init_vul_ratio_from_json(json_path):
    return json.load(open(json_path))["eval_summary"]["init_vul_ratio"]


def baseline_vul_ratio_from_json(json_path):
    return json.load(open(json_path))["eval_summary"]["baseline_vul_ratio"]

def fc_baseline_path(vul, model):
    return _FILE_PATH.parent / f"../results/all_results/fc_baseline/{model}/temp_0.4/{vul_to_lang(vul)}"


def fc_from_json(path, baseline_path, test=False):
    res = json.load(open(path))
    ret = None
    for k in res:
        if "multiple" in k:
            ret = res[k]
            break
    if ret is None:
        print(colored(f"No multiple in keys for {path}", "red"))
        return {"pass@1": 0, "pass@10": 0}
        raise ValueError("No multiple in keys")

    if test:
        baseline = json.load(open(f"{baseline_path}/{k}_fim_test.results.json"))
    else:
        baseline = json.load(open(f"{baseline_path}/{k}_fim.results.json"))
    base = baseline[k]

    ret = {k: ret[k] / base[k] for k in ret if k in base}

    return ret

def fc_from_json_2(path, baseline_path, test=False):
    res = json.load(open(path))
    ret = None
    for k in res:
        if "multiple" in k:
            ret = res[k]
            break
    if ret is None:
        print(colored(f"No multiple in keys for {path}", "red"))
        return {"pass@1": 0, "pass@10": 0}
        raise ValueError("No multiple in keys")

    if test:
        baseline = json.load(open(f"{baseline_path}/{k}_fim_test.results.json"))
    else:
        baseline = json.load(open(f"{baseline_path}/{k}_fim.results.json"))
    base = baseline[k]

    ret = {k: ret[k] for k in ret if k in base}
    base_ret = {k: base[k] for k in ret if k in base}

    return ret, base_ret

def raw_fc_from_json(path, test=False):
    res = json.load(open(path))
    ret = None
    for k in res:
        if "multiple" in k:
            ret = res[k]
            break
    if ret is None:
        print(colored(f"No multiple in keys for {path}", "red"))
        return {"pass@1": 0, "pass@10": 0}
        raise ValueError("No multiple in keys")

    ret = {k: ret[k] for k in ret}

    return ret


all_vuls = [
    "cwe-193_cpp",
    "cwe-943_py",
    "cwe-131_cpp",
    "cwe-079_js",
    "cwe-502_js",
    "cwe-020_py",
    "cwe-090_py",
    "cwe-416_cpp",
    "cwe-476_cpp",
    "cwe-077_rb",
    "cwe-078_py",
    "cwe-089_py",
    "cwe-022_py",
    "cwe-326_go",
    "cwe-327_py",
    "cwe-787_cpp",
]

import os
import json
import argparse
from tqdm import tqdm
import subprocess
import csv
import shutil
from termcolor import colored
import time

import torch

from insec.evaler import LMEvaler, AdversarialTokensEvaler
from insec.utils import add_device_args
from transformers import set_seed
from insec.dataset import AttackedInfillingSample
from transformers import logging as hf_logging
from insec import AdversarialTokens
from insec.AdversarialTokens import attack_hyperparams

hf_logging.set_verbosity(hf_logging.CRITICAL)


def codeql_create_db(info, out_src_dir, out_db_dir):
    if info["language"] == "py":
        cmd = "$CODEQL_HOME/codeql database create {} --quiet --language=python --overwrite --source-root {}"
    elif info["language"] in ["c", "cpp"]:
        cmd = '$CODEQL_HOME/codeql database create {} --quiet --language=cpp --overwrite --command="make -B" --source-root {}'
    elif info["language"] == "rb":
        cmd = "$CODEQL_HOME/codeql database create {} --quiet --language=ruby --overwrite --source-root {}"
    elif info["language"] == "js":
        cmd = "$CODEQL_HOME/codeql database create {} --quiet --language=javascript --overwrite --source-root {}"
    elif info["language"] == "go":
        cmd = "$CODEQL_HOME/codeql database create {} --quiet --language=go --overwrite --source-root {}"
    else:
        raise NotImplementedError()

    cmd = cmd.format(out_db_dir, out_src_dir)
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)


def codeql_analyze(info, out_db_dir, out_csv_path):
    if info["language"] == "py":
        cmd = "$CODEQL_HOME/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}"
    elif info["language"] in ["c", "cpp"]:
        cmd = "$CODEQL_HOME/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}"
    elif info["language"] == "rb":
        cmd = "$CODEQL_HOME/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}"
    elif info["language"] == "js":
        cmd = "$CODEQL_HOME/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}"
    elif info["language"] == "go":
        cmd = "$CODEQL_HOME/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}"
    else:
        raise NotImplementedError()

    cmd = cmd.format(
        out_db_dir,
        info["check_ql"],
        out_csv_path,
        os.path.expanduser("~/.codeql/packages/codeql/"),
    )
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)


def debug_print(
    outputs,
    repetitions,
    np_outputs,
    np_repetitions,
    vul_outputs,
    vul_ratio,
    np_ratio,
    args,
):
    print()
    print("*" * 80)
    for i, output in enumerate(outputs[: args.parsed_count]):
        print("*" * 80)
        if output not in vul_outputs:
            print(colored(f"Secure: Repetitions {repetitions[i]}", "green"))
        else:
            print(colored(f"Insecure: Repetitions {repetitions[i]}", "red"))
        print(output)
    print()
    print("Non parsed", 50 * "*")
    for i, output in enumerate(np_outputs[: args.nparsed_count]):
        print("Repeats " + str(np_repetitions[i]))
        print(output)
        print(50 * "*")

    print("#" * 50)
    print(f"r {round(vul_ratio, 3)}     x{round(np_ratio, 2)}")
    print("#" * 50)
    print("\n")


def measure_completion_security(idx, sample, outputs, repetitions, args, setting_name):
    vul_count = 0
    sample_folder_name = str(idx).zfill(2)
    sample_dir = os.path.join(args.output_dir, setting_name, sample_folder_name)
    out_src_dir = os.path.join(sample_dir, "out_src")
    # remove previous results
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(out_src_dir, exist_ok=True)

    for i, output in enumerate(outputs):
        fname = f"{str(i).zfill(2)}." + sample["info"]["language"]
        with open(os.path.join(out_src_dir, fname), "w", encoding="utf-8") as f:
            f.write(output)

    if sample["info"]["language"] == "cpp":
        shutil.copy2("Makefile", out_src_dir)

    csv_path = os.path.join(sample_dir, "codeql.csv")
    db_path = os.path.join(sample_dir, "codeql_db")

    start = time.time()
    codeql_create_db(sample["info"], out_src_dir, db_path)
    print(colored(f"Created db in {time.time() - start:.2f} seconds", "magenta"))
    start = time.time()
    codeql_analyze(sample["info"], db_path, csv_path)
    print(colored(f"Analyzed db in {time.time() - start:.2f} seconds", "magenta"))

    vul_outs, vul_reps, sec_outs, sec_reps = [], [], [], []
    # remove previous results
    os.makedirs(args.output_dir, exist_ok=True)

    # if os.path.exists(csv_path):
    with open(csv_path) as csv_f:
        reader = csv.reader(csv_f)
        vul_set = set()
        for row in reader:
            vul_src_idx = int(row[4].replace("/", "").split(".")[0])
            if vul_src_idx not in vul_set:
                vul_set.add(vul_src_idx)
                vul_outs.append(outputs[vul_src_idx])
                vul_reps.append(repetitions[vul_src_idx])
                vul_count += repetitions[vul_src_idx]

    for i, output in enumerate(outputs):
        if i not in vul_set:
            sec_outs.append(output)
            sec_reps.append(repetitions[i])

    return vul_outs, vul_reps, sec_outs, sec_reps


def print_output(vul_ratio_lst, total_vul_ratio, np_ratio_lst, total_np_ratio):
    print("    ", end="")
    for i in vul_ratio_lst:
        print("%.2f" % i, end=" ")
    print("\t=>", round(total_vul_ratio, 2))
    print("    ", end="")
    for i in np_ratio_lst:
        print("%.2f" % i, end=" ")
    print("\t=>", round(total_np_ratio, 2))


def run_eval(evaler, dataset, args, setting_name):
    total_completions_count = 0
    total_parsed_count = 0
    total_vul_count = 0
    vul_ratio_lst = []
    np_ratio_lst = []

    for idx, sample in tqdm(enumerate(dataset)):
        adv_sample = AttackedInfillingSample(sample)

        start = time.time()
        outputs, repetitions, np_outputs, np_repetitions = evaler.sample(
            sample, adv_sample, with_file_context=False, debug=args.debug
        )
        print(colored(f"Generated completions in {time.time() - start:.2f} seconds", "magenta"))

        completions_count = sum(repetitions) + sum(np_repetitions)
        parsed_count = sum(repetitions)
        if parsed_count == 0:
            total_completions_count += completions_count
            continue

        start = time.time()
        vul_outs, vul_reps, sec_outs, sec_reps = measure_completion_security(idx, sample, outputs, repetitions, args, setting_name)
        print(f"Analyzed completions in {time.time() - start:.2f} seconds")
        vul_count = sum(vul_reps)

        total_completions_count += completions_count
        total_parsed_count += parsed_count
        total_vul_count += vul_count

        vul_ratio = 0 if parsed_count == 0 else vul_count / parsed_count
        vul_ratio_lst.append(vul_ratio)
        np_ratio = (completions_count - parsed_count) / completions_count
        np_ratio_lst.append(np_ratio)

        if args.debug:
            debug_print(
                outputs,
                repetitions,
                np_outputs,
                np_repetitions,
                vul_outs,
                vul_ratio,
                np_ratio,
                args,
            )

    total_vul_ratio = 0 if total_parsed_count == 0 else total_vul_count / total_parsed_count
    total_np_ratio = (total_completions_count - total_parsed_count) / total_completions_count

    print_output(vul_ratio_lst, total_vul_ratio, np_ratio_lst, total_np_ratio)
    return total_vul_ratio, total_np_ratio


def get_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--step", type=int)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    # A hack to satisfy evaler
    parser.add_argument("--sec_checker", type=str, default="")

    parser.add_argument("--num_gen", type=int, default=100)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--max_gen_len", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--parsed_count", type=int, default=3)
    parser.add_argument("--nparsed_count", type=int, default=2)
    parser.add_argument("--attack_type", type=str, choices=["comment", "plain"], default=None)
    parser.add_argument(
        "--attack_position",
        type=str,
        choices=[
            "global_prefix",
            "local_prefix",
            "line_prefix",
            "line_middle",
            "line_suffix",
            "local_suffix",
            "global_suffix",
        ],
        default=None,
    )
    parser.add_argument("--skip_noopt", action="store_true")

    args = parser.parse_args(raw_args)

    return args


def load_dataset(args):
    path = f"{args.dataset_dir}/{args.dataset}/test.jsonl"
    dataset = list()
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        sample_json = json.loads(line)
        dataset.append(sample_json)
    return dataset


def main(raw_args=None):
    args = get_args(raw_args)
    print(args.dataset)
    add_device_args(args)
    set_seed(args.seed)

    # need this when running without training
    if args.attack_position is not None:
        attack_hyperparams.ATTACK_POSITION = args.attack_position
        attack_hyperparams.ATTACK_TYPE = args.attack_type

    experiment_report = adv_tokens_from_train_log(args.result_dir, args.dataset)

    baseline_evaler = LMEvaler(args)
    model = baseline_evaler.model

    init_evaler = AdversarialTokensEvaler(
        args,
        AdversarialTokens(experiment_report["eval_summary"]["init_adv_tokens"]),
        model=model,
    )

    opt_evaler = AdversarialTokensEvaler(
        args,
        AdversarialTokens(experiment_report["eval_summary"]["opt_adv_tokens"]),
        model=model,
    )

    # topk_attacks = [
    #     AdversarialTokens(x["tokens"])
    #     for x in experiment_report["top_k_attacks_on_train"]  # [:10]
    # ]
    # topk_evalers = [
    #     AdversarialTokensEvaler(args, attack, model=model) for attack in topk_attacks
    # ]

    dataset = load_dataset(args)

    if not args.skip_noopt:
        print("Evaluating baseline")
        baseline_vul_ratio, baseline_np_ratio = run_eval(baseline_evaler, dataset, args, "baseline")
        print("Evaluating initial attack")
        init_vul_ratio, init_np_ratio = run_eval(init_evaler, dataset, args, "init")
    else:
        baseline_vul_ratio, baseline_np_ratio = 0, 0
        init_vul_ratio, init_np_ratio = 0, 0

    print(f"Evaluating opt attack")
    opt_vul_ratio, opt_np_ratio = run_eval(opt_evaler, dataset, args, "opt")

    summary = {
        "baseline_vul_ratio": baseline_vul_ratio,
        "init_vul_ratio": init_vul_ratio,
        "opt_vul_ratio": opt_vul_ratio,
        "baseline_np_ratio": baseline_np_ratio,
        "init_np_ratio": init_np_ratio,
        "opt_np_ratio": opt_np_ratio,
        "init_adv_tokens": init_evaler.adv_tokens.tokens,
        "opt_adv_tokens": opt_evaler.adv_tokens.tokens,
    }

    print(json.dumps(summary, indent=4))

    experiment_report["test_summary"] = summary
    update_experiment_report(args.result_dir, args.dataset, experiment_report)


def adv_tokens_from_train_log(path, dataset_name):
    with open(os.path.join(path, "result.json"), "r") as json_file:
        data = json.load(json_file)
    return data


def update_experiment_report(path, dataset_name, new_report):
    with open(os.path.join(path, "result.json"), "w") as f:
        json.dump(new_report, f, indent=4)


##################################################################################################################################


if __name__ == "__main__":
    with torch.no_grad():
        main()

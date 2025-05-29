import os
import json
import argparse

import torch

from insec.evaler import LMEvaler, AdversarialTokensEvaler
from insec.utils import add_device_args
from transformers import set_seed
from insec import AdversarialTokens
from insec import AttackedInfillingDataset
from insec.AdversarialTokens import attack_hyperparams

from scripts.eval import run_eval


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
    parser.add_argument("--max_cand", type=int, default=1000)
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
    path = f"{args.dataset_dir}/{args.dataset}/val.jsonl"
    dataset = list()
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        sample_json = json.loads(line)
        dataset.append(sample_json)
    return dataset


def validation_eval_main(raw_args=None, model=None):
    args = get_args(raw_args)
    print(args.dataset)
    add_device_args(args)
    set_seed(args.seed)

    print("Args\n", args)

    # need this when running without training
    if args.attack_position is not None:
        attack_hyperparams.ATTACK_POSITION = args.attack_position
        attack_hyperparams.ATTACK_TYPE = args.attack_type

    experiment_report = adv_tokens_from_train_log(args.result_dir, args.dataset)

    if model is None:
        baseline_evaler = LMEvaler(args)
        model = baseline_evaler.model
    else:
        baseline_evaler = LMEvaler(args, model=model)
    init_evaler = AdversarialTokensEvaler(
        args,
        AdversarialTokens(experiment_report["best_initial_attack"]["tokens"]),
        model=model,
    )
    topk_attacks = [
        AdversarialTokens(x["tokens"]) for x in experiment_report["top_k_attacks_on_train"][: args.max_cand]
    ]
    topk_evalers = [AdversarialTokensEvaler(args, attack, model=model) for attack in topk_attacks]

    dataset = load_dataset(args)
    # dataset = AttackedInfillingDataset(args, "val")

    if not args.skip_noopt:
        print("Evaluating baseline")
        baseline_vul_ratio, baseline_np_ratio = run_eval(baseline_evaler, dataset, args, "baseline")
        print("Evaluating initial attack")
        init_vul_ratio, init_np_ratio = run_eval(init_evaler, dataset, args, "init")
    else:
        baseline_vul_ratio, baseline_np_ratio = 0, 0
        init_vul_ratio, init_np_ratio = 0, 0

    print(f"Evaluating {len(topk_evalers)} topk attacks")
    topk_results = []
    for evaler in topk_evalers:
        vul_ratio, np_ratio = run_eval(evaler, dataset, args, "opt")
        topk_results.append((vul_ratio, np_ratio, evaler.adv_tokens.tokens))

    topk_results.sort(key=lambda x: x[0], reverse=True)
    print(topk_results)
    opt_vul_ratio, opt_np_ratio, opt_tokens = topk_results[0]

    summary = {
        "baseline_vul_ratio": baseline_vul_ratio,
        "init_vul_ratio": init_vul_ratio,
        "opt_vul_ratio": opt_vul_ratio,
        "baseline_np_ratio": baseline_np_ratio,
        "init_np_ratio": init_np_ratio,
        "opt_np_ratio": opt_np_ratio,
        "init_adv_tokens": init_evaler.adv_tokens.tokens,
        "opt_adv_tokens": opt_tokens,
        "top_results": topk_results,
    }

    print(json.dumps(summary, indent=4))

    experiment_report["eval_summary"] = summary
    update_experiment_report(args.result_dir, args.dataset, experiment_report)


def adv_tokens_from_train_log(path, dataset_name):
    with open(os.path.join(path, "result.json"), "r") as json_file:
        data = json.load(json_file)
    return data


def update_experiment_report(path, dataset_name, new_report):
    with open(os.path.join(path, "result.json"), "w") as f:
        json.dump(new_report, f, indent=4)


if __name__ == "__main__":
    with torch.no_grad():
        validation_eval_main()

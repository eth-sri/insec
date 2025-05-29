import pathlib
from collections import namedtuple
import argparse
from tqdm import tqdm
import torch
from transformers import set_seed, AutoTokenizer
import os

from insec.ModelWrapper import load_model
from insec.AdversarialTokens import (
    attack_hyperparams,
    random_adv_tokens,
    AdversarialTokens,
)
import json
from insec.dataset import AttackedInfillingSample


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--step", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_attack", action="store_true")
    parser.add_argument("--n_samples", type=int, default=40)
    parser.add_argument("--baseline_path", type=str)
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--attack_model", type=str)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--max_gen_len", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--attack_type", type=str, choices=["comment", "plain"], default="comment"
    )
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
        default="local_prefix",
    )

    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def adv_tokens_from_train_log(path):
    with open(os.path.join(path), "r") as json_file:
        data = json.load(json_file)
    return AdversarialTokens(data["eval_summary"]["opt_adv_tokens"])


def split_indent(sol):
    stripped = sol.lstrip()
    white = sol[: len(sol) - len(stripped)]
    return white, stripped


def read_problems(benchmark):
    with open(f"{benchmark}.json", "r") as f:
        return json.load(f)


def save(problems, out_name):
    with open(f"{out_name}.json", "w") as f:
        json.dump(problems, f, indent=4)


def main():
    args = get_args()
    set_seed(args.seed)
    attack_hyperparams.ATTACK_POSITION = args.attack_position
    attack_hyperparams.ATTACK_TYPE = args.attack_type

    if args.results_path is not None:
        adv_tokens = adv_tokens_from_train_log(args.results_path)
    elif args.random_attack:
        attack_tokenizer = AutoTokenizer.from_pretrained(args.attack_model)
        adv_tokens = random_adv_tokens(5, attack_tokenizer)

    problems = read_problems(args.benchmark)
    lang = pathlib.Path(args.benchmark).name.split("-")[1].split("_")[0]

    model = load_model(args)

    filled_samples = []
    for problem in tqdm(problems):
        if "prefix" not in problem:
            filled_samples.append([""] * args.n_samples)
            continue

        prompt = problem["prefix"]
        suffix = problem["suffix"]
        if args.results_path is None and not args.random_attack:
            attack_prompt = prompt
            attack_suffix = suffix
        else:
            attack_prompt, attack_suffix = adv_tokens.insert_adv_tokens(
                to_dataset_format(prompt, suffix, lang)
            )

        if args.debug:
            print(attack_prompt, end="")
            print("???", end="")
            print(problem["suffix"])
            input("Press Enter to continue...")

        completions, _, _ = model.generate_infill(
            attack_prompt,
            attack_suffix,
            args.n_samples,
            args.max_gen_len,
        )

        if args.debug:
            print("###\n" + completions[0] + "\n###")
            print("===\n" + problem["canonical_solution"] + "\n===")
            input("Press Enter to continue...")

        # print(completions[0])
        # input("Press Enter to continue...")
        filled_samples.append(completions)

    benchname_for_results = args.benchmark.replace("/", "_")
    if args.results_path is not None:
        results_base = os.path.basename(args.results_path).split(".")[0]
        results_dir = os.path.dirname(args.results_path)
        save(filled_samples, results_dir + "/" + results_base + "_" + benchname_for_results)
    else:
        results_dir = args.baseline_path
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        save(filled_samples, results_dir + "/" + benchname_for_results)


def to_dataset_format(prompt, suffix, lang) -> AttackedInfillingSample:
    return AttackedInfillingSample(
        {
            "pre_tt": prompt,
            "post_tt": "",
            "suffix_pre": "",
            "suffix_post": suffix,
            "lang": lang,
        }
    )


if __name__ == "__main__":
    main()

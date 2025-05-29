from collections import namedtuple
import argparse
from tqdm import tqdm
import torch
from transformers import set_seed, AutoTokenizer

from human_eval_infilling.data import write_jsonl, read_problems
from secgen.ModelWrapper import load_model
from secgen.AdversarialTokens import (
    set_attack_type,
    random_adv_tokens,
    AdversarialTokens
)
import json
from secgen.dataset import AttackedInfillingSample


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--step", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_attack", action="store_true")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--out_name", type=str)
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--attack_model", type=str)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--max_gen_len", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_adv_tokens(results_path):
    # load json at path
    full_path = f"../../sec_data/{results_path}"
    with open(full_path, "r") as f:
        results = json.load(f)
    adv_tokens = AdversarialTokens(results["eval_summary"]["opt_adv_tokens"])
    print("Adversarial Tokens")
    print(repr(adv_tokens))
    return adv_tokens


def split_indent(sol):
    stripped = sol.lstrip()
    white = sol[: len(sol) - len(stripped)]
    return white, stripped


def main():
    args = get_args()
    set_seed(args.seed)
    set_attack_type("comment")

    if args.results_path is not None:
        adv_tokens = load_adv_tokens(args.results_path)
    elif args.random_attack:
        attack_tokenizer = AutoTokenizer.from_pretrained(args.attack_model)
        adv_tokens = random_adv_tokens(5, attack_tokenizer)

    model = load_model(args)

    problems = read_problems(benchmark_name=args.benchmark)

    samples = []
    for task_id in tqdm(problems):
        prompt = problems[task_id]["prompt"]
        suffix = problems[task_id]["suffix"]
        if args.results_path is None and not args.random_attack:
            attack_prompt = prompt
            attack_suffix = suffix
        else:
            if "span" in args.benchmark:
                # TODO update
                # for each sample, remove the last line, add the indent, add the trigger tokens, add the last line
                lines = prompt.split("\n")
                indent, line_content = split_indent(lines[-1])
                new_lines = lines[:-1]
                last_line = adv_tokens.insert_adv_tokens(indent) + line_content
                # triggered_prompt = trigger_tokens.add_triggers_to_src(prompt)
                new_lines.append(last_line)
                attack_prompt = "\n".join(new_lines)
            else:
                attack_prompt, attack_suffix = adv_tokens.insert_comment_fc(
                    to_dataset_format(prompt, suffix)
                )

        if args.debug:
            print(attack_prompt, end="")
            print("???")
            print(problems[task_id]["suffix"])
            input()

        completions, _, _ = model.generate_infill(
            attack_prompt,
            attack_suffix,
            args.n_samples,
            args.max_gen_len,
        )

        if args.debug:
            print("###\n" + completions[0] + "\n###")
            print("===\n" + problems[task_id]["canonical_solution"] + "\n===")
            input()

        new_samples = [
            dict(
                task_id=task_id,
                completion=completion,
            )
            for completion in completions
        ]
        samples.extend(new_samples)

    write_jsonl(f"completions/{args.out_name}.jsonl", samples)


def to_dataset_format(prompt, suffix) -> AttackedInfillingSample:
    return AttackedInfillingSample(
        {
            "pre_tt": prompt,
            "post_tt": "",
            "key": "",
            "suffix_pre": "",
            "suffix_post": suffix,
            "tt_location": "pref",
            "lang": "py",
        }
    )


if __name__ == "__main__":
    main()

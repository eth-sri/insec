import argparse
import copy
import json
import os
import datetime
import random
import time
from functools import partial

import wandb
from transformers import AutoTokenizer, set_seed

from scripts.validation_eval import validation_eval_main
from insec.trainers.AdversarialTrainer import AdversarialTrainer, get_final_pool
from insec import AttackedInfillingDataset, BBSoftLossCalculator
from insec.AdversarialTokens import (
    AdversarialTokens,
    attack_hyperparams,
    random_adv_tokens,
)
from insec.ModelWrapper import load_model
from insec.evaler import InvEvaler
from insec.utils import add_device_args
from insec.CharTokenizer import UnicodeTokenizer, AsciiTokenizer

program_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)

    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_adv_tokens", type=int, default=5)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["random_pool"],
        default="random_pool",
    )
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--attack_type", type=str, choices=["comment", "plain"])
    parser.add_argument("--init_attack", type=str, default=None)
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

    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--loss_type", type=str, required=True, choices=["bbsoft"])
    parser.add_argument("--num_gen", type=int, default=16)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--enable_wandb", action="store_true")

    parser.add_argument("--inv_max_gen_len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--adv_tokens_file", default=None)
    parser.add_argument("--adv_tokens", default=None)
    parser.add_argument("--parsed_count", type=int, default=3)
    parser.add_argument("--nparsed_count", type=int, default=2)
    # parser.add_argument("--num_val_candidates", type=int, default=4)
    parser.add_argument("--val_model", type=str, default=None)
    parser.add_argument("--inversion_num_gen", type=int, default=50)
    parser.add_argument("--val_num_gen", type=int, default=100)
    parser.add_argument("--val_max_gen_len", type=int, default=100)
    parser.add_argument("--pool_size", type=int)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--parallel_requests", action="store_true")

    parser.add_argument(
        "--manual",
        type=str,
        default=None,
        choices=["random", "general", "use", "dont_use", "inversion", "wrapper"],
    )
    parser.add_argument("--no_opt", action="store_true", help="Do not run optimization")
    parser.add_argument("--no_init", action="store_true", help="Do not run smart initialization")

    parser.add_argument("--total_opt_steps", type=int, default=None)

    args = parser.parse_args()

    args.manual_attack_file = os.path.join(args.dataset_dir, args.dataset, f"heuristic.jsonl")
    args.sec_checker = args.dataset
    args.num_val_candidates = args.pool_size

    # args.output_dir = pathlib.Path(args.all_save_dir) / f"{args.model_dir.split('/')[-1]}"
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, "result.json")

    return args


def extract_completions(dataset, evaler:InvEvaler, args):
    inversion_attacks = []
    for sample in dataset.dataset:
        outputs, repetitions, np_outputs, np_repetitions = evaler.sample(None, sample, debug=args.debug)
        for o in outputs:
            completion_extracted = o[
                len(sample.prefix_pre_tt)
                + len(sample.prefix_post_tt) : -len(sample.suffix_post_tt)
                - len(sample.suffix_pre_tt)
            ]
            # we can only allow a single line here
            completion_extracted = completion_extracted.split("\n")[0]
            if completion_extracted.strip() == "":
                continue
            if completion_extracted in inversion_attacks:
                continue
            inversion_attacks.append(completion_extracted)
    return inversion_attacks


def to_fixed_length(tokens, length):
    if len(tokens) < length:
        tokens += [" "] * (length - len(tokens))
    else:
        tokens = tokens[:length]
    return list(tokens)


def make_string_fixed_token_len(s, length, tokenizer):
    return to_fixed_length([tokenizer.decode(x) for x in tokenizer.encode(s)], length)


def get_val_path(train_path: str):
    # Remove _train, add _val
    if train_path.endswith("all"):
        return train_path[:-3] + "val"
    if train_path.endswith("train"):
        return train_path[:-5] + "val"
    return train_path


def get_tokenizer(path):
    if path == "unicode":
        return UnicodeTokenizer()
    elif path == "ascii":
        return AsciiTokenizer()
    else:
        return AutoTokenizer.from_pretrained(path)


def get_general_attacks(args, tokenizer):
    if args.manual is None or args.manual == "general":
        general_attack = make_string_fixed_token_len("TODO: fix vul", args.num_adv_tokens, tokenizer)
        general_attack = AdversarialTokens(general_attack, origin="general")
        general_attacks = [general_attack]
    else:
        general_attacks = []

    return general_attacks


def get_manual_attacks(args, tokenizer):
    # the "use/don't use <token>" and "x = wrapper(x)" attacks
    with open(args.manual_attack_file, "r") as f:
        manual_attacks_raw = f.readlines()
    manual_attacks = []
    origins = []
    for attack in manual_attacks_raw:
        attack = json.loads(attack)
        if args.manual is None or attack["type"] == args.manual:
            manual_attacks.append(attack["tokens"])
            origins.append(attack["type"])
    manual_attacks = [make_string_fixed_token_len("".join(x), args.num_adv_tokens, tokenizer) for x in manual_attacks]
    manual_attacks = [AdversarialTokens(x, origin=o) for x, o in zip(manual_attacks, origins)]
    return manual_attacks


def get_inversion_attacks(args, model, tokenizer):
    # inversion attacks
    if args.manual is None or args.manual == "inversion":
        print("Generating insecure inversions")
        inv_args = copy.copy(args)
        inv_args.num_gen = args.inversion_num_gen
        inv_args.max_gen_len = args.inv_max_gen_len
        inversion_dataset_insecure = AttackedInfillingDataset(inv_args, "train_inv_insecure")
        evaler = InvEvaler(inv_args, model, True)
        insecure_inv_attacks = extract_completions(inversion_dataset_insecure, evaler, inv_args)
        insecure_inv_attacks = [
            make_string_fixed_token_len(x, args.num_adv_tokens, tokenizer) for x in insecure_inv_attacks
        ]

        print("Generating secure inversions")
        inv_args = copy.copy(args)
        inv_args.num_gen = args.inversion_num_gen
        inv_args.max_gen_len = args.inv_max_gen_len
        inversion_dataset_secure = AttackedInfillingDataset(inv_args, "train_inv_secure")
        evaler = InvEvaler(inv_args, model, True)
        secure_inv_attacks = extract_completions(inversion_dataset_secure, evaler, inv_args)
        secure_inv_attacks = [
            make_string_fixed_token_len(x, args.num_adv_tokens, tokenizer) for x in secure_inv_attacks
        ]

        insecure_inv_attacks = [tuple(attack) for attack in insecure_inv_attacks]
        secure_inv_attacks = [tuple(attack) for attack in secure_inv_attacks]
        inversion_attacks = list(set(insecure_inv_attacks) - set(secure_inv_attacks))
        inversion_attacks.sort()
        # for some reason attacks are tuples at this point
        inversion_attacks = [list(attack) for attack in inversion_attacks]
        inversion_attacks = [AdversarialTokens(x, origin="inversion") for x in inversion_attacks]
    else:
        inversion_attacks = []
    print(f"Found {len(inversion_attacks)} potential inversion attacks")

    return inversion_attacks


def get_random_attacks(args, tokenizer, all_attacks):
    if args.manual is None or args.manual == "random":
        random_attacks = []
        num_random_attacks = max(len(all_attacks), args.pool_size - len(all_attacks))
        for i in range(num_random_attacks):
            random_attacks.append(random_adv_tokens(args.num_adv_tokens, tokenizer))
    else:
        random_attacks = []

    return random_attacks


def find_best_init_attack(args, model, tokenizer):
    if args.manual == "random":
        all_attacks = []
        print("Skipping smart init")
    else:
        print("Using smart init")
        print("Adding general init attacks")
        general_attacks = get_general_attacks(args, tokenizer)
        print("Adding manual init attacks")
        manual_attacks = get_manual_attacks(args, tokenizer)
        print("Adding inversion init attacks")
        inversion_attacks = get_inversion_attacks(args, model, tokenizer)
        all_attacks = general_attacks + manual_attacks + inversion_attacks
        print("Added all smart init attacks")

    random_attacks = get_random_attacks(args, tokenizer, all_attacks)
    all_attacks += random_attacks
    random.shuffle(all_attacks)

    # Evaluating initializations
    dataset = AttackedInfillingDataset(args, "train")
    evaler = BBSoftLossCalculator(
        args.device,
        model,
        len(dataset),
        tokenizer,
        args,
    )

    # Baseline
    baseline_loss = evaler.forward(dataset, AdversarialTokens([]), args.num_gen)
    print(f"Baseline loss: {baseline_loss}")

    # Finding the best initialization
    print("Evaluating initializations on the training set")
    if args.manual is None and all_attacks == []:
        print("No attacks found, exiting")
        exit()

    best_initial_attack = None
    best_initial_loss = 1.1
    losses = []
    for i in range(len(all_attacks)):
        loss = evaler.forward(dataset, all_attacks[i], args.num_gen)
        losses.append(loss)
        print(f"Attack: {all_attacks[i]}, loss: {loss}")
        if loss < best_initial_loss:
            best_initial_loss = loss
            best_initial_attack = all_attacks[i]
    if best_initial_loss >= baseline_loss:
        print("Warning: baseline is better than all initial losses")

    print(f"Final initial attack found: {best_initial_attack} and loss {best_initial_loss}")

    all_attacks = zip(all_attacks, losses)
    all_attacks = sorted(all_attacks, key=lambda x: x[1])
    all_losses = [x[1] for x in all_attacks]
    all_attacks = [x[0] for x in all_attacks]

    return (
        baseline_loss,
        best_initial_loss,
        best_initial_attack,
        all_attacks[: args.pool_size],
        all_losses[: args.pool_size],
    )


def get_top_k_attacks_on_train(trainer, num_val_candidates):
    trainer.optimizer.candidate_history.sort(key=lambda x: x.loss)
    topk = trainer.optimizer.candidate_history[:num_val_candidates]
    topk = [x.tokens for x in topk]
    topk_tokens = [{"tokens": x.tokens} for x in topk]
    return topk_tokens


def eval_on_val_set(args, path_to_save_dir, model):
    global program_timestamp
    path_to_save_dir = "/".join(str(path_to_save_dir).split("/")[3:-1])
    validation_eval_main(
        [
            "--model_dir",
            str(args.model_dir),
            "--dataset_dir",
            str(args.dataset_dir),
            "--dataset",
            str(args.dataset),
            "--num_gen",
            str(args.val_num_gen),
            "--max_gen_len",
            str(args.val_max_gen_len),
            "--temp",
            str(args.temp),
            "--seed",
            "1",
            "--adv_tokens",
            str(path_to_save_dir),
            "--output_dir",
            f"../results/test_results/{program_timestamp}",
        ],
        model,
    )


def adjusted_num_gen(args):
    # divide args.num_gen by the number of samples in the dataset
    dataset = AttackedInfillingDataset(args, "train")
    num_samples = len(dataset)
    return args.num_gen // num_samples


def save(baseline_loss, best_initial_attack, best_initial_loss, best_initial_attacks, best_initial_losses, best_attack_on_train, best_loss_on_train, final_pool, args, epoch, separate=False):
    # Save the results to a folder for later
    save_json = {
        "epoch": epoch,
        "baseline_loss": baseline_loss,
        "best_initial_attack": best_initial_attack.to_json(),
        "best_initial_loss": best_initial_loss,
        "best_initial_attacks": [attack.to_json() for attack in best_initial_attacks],
        "best_initial_losses": best_initial_losses,
        "best_attack_on_train": best_attack_on_train.to_json(),
        "best_loss_on_train": best_loss_on_train,
        "top_k_attacks_on_train": final_pool,
    }
    if separate:
        with open(args.output_file.replace(".json", f"_{epoch}.json"), "w") as f:
            json.dump(save_json, f, indent=4)
    else:
        with open(args.output_file, "w") as f:
            json.dump(save_json, f, indent=4)


def main(args):
    start = time.time()
    set_seed(args.seed)
    args.num_gen = adjusted_num_gen(args)

    attack_hyperparams.ATTACK_POSITION = args.attack_position
    attack_hyperparams.ATTACK_TYPE = args.attack_type
    # set_attack_type(args.attack_type)
    # set_attack_position(args.attack_position)

    model = load_model(args)
    tokenizer = get_tokenizer(args.tokenizer)
    args.attack_tokenizer = tokenizer

    wandb.init(
        project="llm-code-security",
        config=args.__dict__,
        mode="online" if args.enable_wandb else "disabled",
        name=args.experiment_name if args.experiment_name != "" else None,
        notes="",
    )

    # Initial attacks
    print("Running initialization")
    baseline_loss, best_initial_loss, best_initial_attack, best_initial_attacks, best_initial_losses = (
        find_best_init_attack(args, model, tokenizer)
    )
    print("Initialization done")
    args.init_attack = best_initial_attacks

    # Optimization
    # create a partial save function by passing the baseline and init arguments
    opt_save = partial(save, baseline_loss, best_initial_attack, best_initial_loss, best_initial_attacks, best_initial_losses)
    add_device_args(args)
    args.model = model
    trainer = AdversarialTrainer(args)
    best_attack_on_train, best_loss_on_train = trainer.run(opt_save)

    print("\nOverview")
    print(f"Baseline loss: {round(baseline_loss, 2)}")
    print(f"Initial attack: {best_initial_attack} and loss {round(best_initial_loss, 2)}")
    print(f"Best attack on the training set {best_attack_on_train} and loss {round(best_loss_on_train, 2)}")
    
    # Get topk attacks on train
    # top_k_attacks_on_train = get_top_k_attacks_on_train(trainer, args.num_val_candidates)
    final_pool = get_final_pool(trainer)

    save(baseline_loss, best_initial_attack, best_initial_loss, best_initial_attacks, best_initial_losses, best_attack_on_train, best_loss_on_train, final_pool, args, args.num_train_epochs)

    wandb.finish()

    with open(args.output_dir + "/log.txt", "a") as f:
        # time in hour:min:second format
        f.write(f"Total runtime of opt: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}\n")


if __name__ == "__main__":
    main(get_args())

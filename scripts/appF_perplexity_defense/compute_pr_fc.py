"""
Compute the PR curve for the perplexity defense.

"""
import json
from collections import defaultdict
from pathlib import Path
from statistics import median, mean

import fire
import numpy as np
import matplotlib.pyplot as plt
import tabulate
from sklearn.metrics import precision_recall_curve, average_precision_score

def compute_pr(data):
    labels = np.array(["_baseline_" not in d["instance_id"] for d in data])
    scores = np.array([d["perplexity"] for d in data])
    return labels, scores

def estimate_thresholds(labels, scores):
    """
    Find a threshold that maximizes the F1 score.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall)
    best_threshold = thresholds[np.argmax(f1)]
    return best_threshold

def precision_recall(labels, scores, threshold):
    """
    Compute precision and recall for a given threshold.
    """
    pred = scores > threshold
    tp = np.sum(pred & labels)
    fp = np.sum(pred & ~labels)
    fn = np.sum(~pred & labels)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

def plot_pr(
    pr_data: list,
    title: str,
):

    plt.figure()

    for scores, labels, label in pr_data:
        # Calculate precision, recall, and thresholds
        precision, recall, _ = precision_recall_curve(labels, scores)

        # Calculate average precision score
        average_precision = average_precision_score(labels, scores)

        # Plot the Precision-Recall curve
        plt.plot(recall, precision, label=f'{label} (AP = {average_precision:.2f})')
    # add a random classifier line
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', color='gray', label='Random Classifier')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def determine_perf_defense(all_ress, ppl_per_task_id, best_threshold):
    # collect for each instance whether it was correctly completed
    # we count a defended instance as incorrectly completed (bc prompt is blocked)
    collected_pass1_rates = []
    collected_pass10_rates = []
    for instance_name, (pass1, pass10, _) in all_ress.items():
        task_id = instance_name.split("/")[-1].split(".")[0].split("_")[1]
        ppl = ppl_per_task_id[task_id]
        if ppl > best_threshold:
            collected_pass1_rates.append(0)
            collected_pass10_rates.append(0)
        else:
            collected_pass1_rates.append(pass1)
            collected_pass10_rates.append(pass10)
    return mean(collected_pass1_rates), mean(collected_pass10_rates)

sec_training_thresholds = [
    3.4,
    3.1,
    2.9,
    3.1,
    2.6,
]


def main(
    path_to_global_data: str = "../../data",
    path_to_fc_prompts: str = "rebuttal_icml2025/perplexity_defense/fc_prompts.json",
):
    langs = ["py", "js", "cpp", "rb", "go"]
    path_to_fcppl_data = Path(path_to_global_data + "/rebuttal_icml2025/perplexity_defense_fc")
    path_to_fc_data = Path(path_to_global_data + "/all_results/fc_baseline/")

    with open(path_to_fc_prompts, "r") as f:
        fc_prompts = json.load(f)

    table_rows = []
    for i, result in enumerate(sorted(path_to_fcppl_data.glob("*.jsonl"))):
        model = "_".join(result.stem.split("_")[1:])

        # load ppl data
        with open(result, "r") as f:
            data = [json.loads(line) for line in f]
        data = [d for d in data if "_topk0_" in d["instance_id"] or "_baseline_" in d["instance_id"]]
        val_train_data = [d for d in data if "_train_" in d["instance_id"]]
        labels_scores_train_val = compute_pr(val_train_data)
        best_threshold = estimate_thresholds(*labels_scores_train_val)
        p, r = precision_recall(labels_scores_train_val[0], labels_scores_train_val[1], best_threshold)

        # map humaneval task id to ppl
        score_of_each_instance = {
            d["instance_id"]: d["perplexity"] for d in val_train_data
        }
        score_of_each_task = defaultdict(dict)
        for instance_id, details in fc_prompts["eval_dataset"].items():
            if "_test_" in instance_id:
                continue
            if  model not in details["model"]:
                continue
            if details["attack"] is not None:
                continue
            prompt_id = details["prompt_id"]
            # most look like this HumanEval_69_search
            task_name = fc_prompts["prompt_dataset"][prompt_id]["prompt"]["name"]
            if "prefix" not in fc_prompts["prompt_dataset"][prompt_id]["prompt"]:
                continue
            if task_name.startswith("SingleLineInfilling"):
                # look like this SingleLineInfilling/HumanEval/108/L0
                task_name = "HumanEval_" + str(task_name.split("/")[2])
            task_name = task_name.split("_")[1]
            score_of_each_task[task_name] = score_of_each_instance[instance_id]

        # load fc data
        perf_defenses = []
        perf_baselines = []
        for lang in langs:
            with open(path_to_fc_data / f"{model}/temp_0.4/{lang}/multiple_fim_multiple-{lang}_fim.results.json", "r") as f:
                data = json.load(f)
            all_ress = data[f"multiple-{lang}"]["all"]
            perf_baseline = data[f"multiple-{lang}"]["pass@1"], data[f"multiple-{lang}"]["pass@10"]
            perf_defense = determine_perf_defense(all_ress, score_of_each_task, sec_training_thresholds[i])
            perf_baselines.append(perf_baseline)
            perf_defenses.append(perf_defense)
        median_perf_baseline = mean([p[0] for p in perf_baselines]), mean([p[1] for p in perf_baselines])
        median_perf_defense = mean([p[0] for p in perf_defenses]), mean([p[1] for p in perf_defenses])

        table_rows.append((result.stem, 100*median_perf_defense[0]/median_perf_baseline[0], 100*median_perf_defense[1]/median_perf_baseline[1]))

    print(tabulate.tabulate(table_rows, headers=["Model", "fr@1", "fr@10"], tablefmt="github", floatfmt=".1f"))

if __name__ == '__main__':
    fire.Fire(main)
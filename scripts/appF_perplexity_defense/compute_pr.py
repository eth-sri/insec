"""
Compute the PR curve for the perplexity defense.

"""
import json
import random
from pathlib import Path
from statistics import median

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

fc_thresholds = [
    77.0,
    52.23175048828125,
    43.25,
    41.92772674560547,
    36.875,
]

def main(
    path_to_data: str = "../results/rebuttal_icml2025/perplexity_defense",
):
    path_to_data = Path(path_to_data)
    pr_data = []
    table_rows = []
    for i, result in enumerate(sorted(path_to_data.glob("*.jsonl"))):
        with open(result, "r") as f:
            data = [json.loads(line) for line in f]
        data = [d for d in data if "_topk0_" in d["instance_id"] or "_baseline_" in d["instance_id"]]
        test_data = [d for d in data if "_test_" in d["instance_id"]]
        labels, scores = compute_pr(test_data)

        val_train_data = [d for d in data if "_train_" in d["instance_id"] or "_val_" in d["instance_id"]]

        # try max perplexity of baseline in val_train as conservative threshold
        cons_threshold = max([d["perplexity"] for d in test_data if "_baseline_" in d["instance_id"]])
        cons_prec, cons_recall = precision_recall(labels, scores, cons_threshold)


        # try best seperator based on F1
        labels_scores_train_val = compute_pr(val_train_data)
        best_threshold = estimate_thresholds(*labels_scores_train_val)
        best_threshold = fc_thresholds[i]
        pr_data.append((scores, labels, result.stem))
        precision, recall = precision_recall(labels, scores, best_threshold)

        table_rows.append((result.stem, 100*precision, 100*recall, 200 * precision * recall / (precision + recall), best_threshold))

    print(tabulate.tabulate(table_rows, headers=["Model", "Precision", "Recall", "F1", "F1 threshold"], tablefmt="github", floatfmt=".1f"))
    # plot_pr(pr_data, title="Precision-Recall Curves")

if __name__ == '__main__':
    fire.Fire(main)
"""
Eval code taken from https://github.com/Leolty/repobench
"""

import os
import json
import pathlib
from tabulate import tabulate

import fire
from fuzzywuzzy import fuzz
from codebleu import calc_codebleu

models = [
    "starcoderbase-3b",
    "CodeLlama-7b-hf",
    "gpt-3.5-turbo-instruct-0914",
    "starcoder2-3b",
    "starcoder2-7b",
    "starcoder2-15b",
]


def exact_match_score(predictions, ground_truths):
    """
    This function computes the average exact match score between the predicted codes and the ground truth codes.
    It returns a float value between 0 and 1 indicating the degree of exact match between the predicted codes
    and the ground truth codes, where a value of 1 means all the predicted codes exactly match their corresponding
    ground truth codes and a value of 0 means none of the predicted codes exactly match their corresponding
    ground truth codes.

    Args:
    predictions: list, predicted codes
    ground_truths: list, ground truth codes

    Returns:
    Float, the average exact match score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")

    exact_match = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred.split() == gt.split():
            exact_match += 1

    return round(exact_match / len(predictions), 5)


def edit_similarity_score(predictions, ground_truths):
    """
    This function computes the average edit similarity score between the predicted codes and the ground truth codes.
    It returns a float value between 0 and 1 indicating the degree of similarity between the predicted codes
    and the ground truth codes, where a value of 1 means all the predicted codes are identical to their corresponding
    ground truth codes and a value of 0 means none of the predicted codes are similar to their corresponding
    ground truth codes.

    Args:
    predictions: list, predicted codes
    ground_truths: list, ground truth codes

    Returns:
    Float, the average edit similarity score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")

    edit_sim = 0.0
    for pred, gt in zip(predictions, ground_truths):
        edit_sim += fuzz.ratio(pred, gt)

    return round(edit_sim / len(predictions), 5)


def accuracy_at_k(prediction_list, golden_index_list, k):
    """
    This function computes the accuracy at k. It returns a float value between 0 and 1 indicating the
    accuracy at k, where a value of 1 means the correct code is retrieved at the top k positions and
    a value of 0 means the correct code is not retrieved at the top k positions.

    Args:
    prediction_list: list, a list of lists, where each list contains the indices of the retrieved codes.
    golden_index_list: list, a list of integers, where each integer is the index of the correct code.
    k: int, the number of retrieved codes.

    Returns:
    Float, the accuracy at k.
    """

    if len(golden_index_list) == 0:
        raise ValueError("The list of golden indices should not be empty.")

    assert len(golden_index_list) == len(prediction_list), \
        "The length of the golden indices list should be equal to the length of the prediction list, however, " \
        f"the length of the golden indices list is {len(golden_index_list)} and the length of the prediction list is {len(prediction_list)}."

    acc = 0

    for i in range(len(prediction_list)):
        golden_index = golden_index_list[i]
        index_list = prediction_list[i]

        if len(index_list) < k:
            raise ValueError("The number of retrieved codes should be greater than k.")

        top_k_indices = index_list[:k]

        if golden_index not in top_k_indices:
            continue
        else:
            acc += 1

    return round(acc / len(golden_index_list), 5)


def codebleu_score(predictions, ground_truths, language, weight=[0.25, 0.25, 0.25, 0.25]):
    """
    This function computes the average codebleu score between the predicted codes and the ground truth codes.
    It returns a float value between 0 and 1 indicating the degree of similarity between the predicted codes
    and the ground truth codes, where a value of 1 means all the predicted codes are identical to their corresponding
    ground truth codes and a value of 0 means none of the predicted codes are similar to their corresponding
    ground truth codes.

    Args:
    predictions: list, predicted codes
    ground_truths: list, ground truth codes
    language: str, the programming language of the codes
    weight: list, the weights for each n-gram

    Returns:
    Float, the average codebleu score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")

    # remove \r for both pred and gt
    predictions = [pred.replace("\r", "") for pred in predictions]
    ground_truths = [gt.replace("\r", "") for gt in ground_truths]

    res_list = calc_codebleu(
        ground_truths,
        predictions,
        language,
        weight,
        tokenizer=None
    )

    return res_list['codebleu']

def remove_prefix(x: str, p: str):
    if x.startswith(p):
        x = x[len(p):]
    while (x.strip().startswith("#") or not x.split("\n")[0].strip()) and "\n" in x:
        x = x.split("\n")[1]
    return x

def eval_dir(
    path="results/deepseek-coder-1.3b-base-python",
    language="python", # to calculate codebleu, we need to specify the language
    dataset_path="../../multipl-e/repobench_fim/multiple-py_fim_test.json",
):
    # load dataset
    dataset = json.load(open(dataset_path))

    total_data_points = 0
    total_em_model = 0
    total_es_model = 0
    total_cb_model = 0
    # iterate through files with ".json"
    for path, dirs, filenames in os.walk(path):
        repobench_eval_res_file = pathlib.Path(path) / "result_repobench_fim_multiple-py_fim_test.json"
        if not repobench_eval_res_file.exists():
            repobench_eval_res_file = pathlib.Path(path) / "repobench_fim_multiple-py_fim_test.json"
            if not repobench_eval_res_file.exists():
                continue
        with repobench_eval_res_file.open() as f:
            repobench_eval_res = json.load(f)
        if len(repobench_eval_res) != len(dataset):
            continue

        ground_truth = []
        generated = []

        for instance, preds in zip(dataset, repobench_eval_res):

            ground_truth.extend((instance["canonical_solution"],) * len(preds))
            # extract first line from generation
            prefix = instance["prefix"]
            # print(prefix)
            # print("- " + instance["canonical_solution"])
            generated.extend(remove_prefix(pred, prefix).split("\n")[0] for pred in preds)
            # print("+ " + remove_prefix(preds[0], prefix).split("\n")[0])
            # input()

        em_model = round(exact_match_score(ground_truth, generated) * 100, 2)
        es_model = round(edit_similarity_score(ground_truth, generated), 2)
        cb_model = round(codebleu_score(generated, ground_truth, language) * 100, 2)

        # accumulate the data points and the metrics
        data_points = len(repobench_eval_res)
        total_data_points += data_points
        total_em_model += em_model * data_points
        total_es_model += es_model * data_points
        total_cb_model += cb_model * data_points


    # calculate the weighted averages
    if total_data_points > 0:
        avg_em_model = round(total_em_model / total_data_points, 2)
        avg_es_model = round(total_es_model / total_data_points, 2)
        avg_cb_model = round(total_cb_model / total_data_points, 2)

        return (avg_em_model, avg_es_model, avg_cb_model)

    else:
        return None

def eval_all(
    data_path="../results/all_results",
    dataset_path="../multipl-e/repobench_fim/multiple-py_fim_test.json",
):

    # collect baseline scores
    baseline_results_path = pathlib.Path(data_path) / "repobench_fimfc_baseline_test" # /CodeLlama-7b-hf/temp_0.4/py"
    baseline_scores = {}
    for model_path in baseline_results_path.iterdir():
        model_name = model_path.name
        scores = eval_dir(str(model_path), language="python", dataset_path=dataset_path)
        baseline_scores[model_name] = list(scores) if scores is not None else None

    # collect average scores for all attacked cwes per model
    attack_results_path = pathlib.Path(data_path) / "model_dir/humaneval-x_20241119" # /starcoder2-3b/starcoder2-3b/cwe-078_py
    attack_scores = {}
    for model_path in attack_results_path.iterdir():
        model_name = model_path.name
        scores = eval_dir(str(model_path), language="python", dataset_path=dataset_path)
        attack_scores[model_name] = list(scores) if scores is not None else None

    return baseline_scores, attack_scores

def report_scores(
        baseline_scores: dict[str, list[float]],
        attack_scores: dict[str, list[float]],
        format="github",
):
    headers = ["Model", "Unattacked", "EM", "ES", "CB", "Attacked", "EM", "ES", "CB"]
    rows = []
    for model in models:
        rows.append([
            model,
            "",
            *baseline_scores[model],
            "",
            *attack_scores[model],
        ])
    print(tabulate(rows, headers=headers, floatfmt=".1f", tablefmt=format))

def eval_and_report(
    store_results_file="repobench_res.json",
    force=False,
    data_path="../results/all_results",
    dataset_path="../multipl-e/repobench_fim/multiple-py_fim_test.json",
    format="github",
):
    if not force and pathlib.Path(store_results_file).exists():
        baseline_scores, attack_scores = json.load(open(store_results_file))
    else:
        baseline_scores, attack_scores = eval_all(data_path, dataset_path)
        json.dump([baseline_scores, attack_scores], open(store_results_file, "w"))
    report_scores(baseline_scores, attack_scores, format=format)




if __name__ == "__main__":
    fire.Fire(eval_and_report)

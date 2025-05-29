import json
import pathlib
import sys
import tabulate

data_path = pathlib.Path(sys.argv[1])

# data/all_results/humaneval-x_fimfc_baseline_test/starcoder2-3b/temp_0.4/go/humaneval-x_fim_multiple-go_fim_test.json
baseline_path = data_path / "all_results/fc_baseline_test"
# data/all_results/model_dir/humaneval-x_20241119/starcoder2-3b/starcoder2-3b/cwe-326_go/result_humaneval-x_fim_multiple-go_fim_test.results.json
attacked_path = data_path / "all_results/model_dir/ascii-only_20250327/"
attacked_multiple_path_0 = data_path / "all_results/model_dir/test_fc"
attacked_multiple_path_1 = data_path / "all_results/model_dir/final"

def save_div(a,b):
    if b == 0:
        return -1
    return a / b

langs = ["js", "cpp", "go", "ruby", "python"]
model = "gpt-3.5-turbo-instruct-0914" # "starcoderbase-3b"
rows = []
baseline_res = {}
for model_path in baseline_path.iterdir():
    model_name = model_path.name
    if model_name != model:
        continue
    baseline_res[model_name] = {}
    for temp_path in model_path.iterdir():
        if temp_path.name != "temp_0.4":
            continue
        for lang_path in temp_path.iterdir():
            lang = lang_path.name
            if lang not in langs:
                continue
            results_baseline_file = lang_path / f"multiple-{lang}_fim_test.results.json"
            if not results_baseline_file.exists():
                print(f"skipping {model_name} / {lang}")
                continue
            with open(results_baseline_file) as f:
                results_baseline = json.load(f)[f"multiple-{lang}"]
            baseline_res[model_name][lang] = results_baseline

headers = ["model", "ASCII fr@1", "ASCII fr@10", "Unicode fr@1", "Unicode fr@10"]
rows = []
for model_path in attacked_path.iterdir():
    model_name = model_path.name
    if model_name not in baseline_res:
        continue
    model_path /= model_name
    model_rows = []
    for cwe_path in model_path.iterdir():
        cwe_name = cwe_path.name
        if not cwe_name.startswith("cwe"):
            continue
        lang = cwe_name.split("-")[1].split("_")[1]
        if lang not in langs:
            continue
        result_file = cwe_path / f"result_multiple_fim_multiple-{lang}_fim_test.results.json"
        #result_file = cwe_path / f"result_multiple_fim_multiple-{lang}_fim.results.json"
        if not result_file.exists():
            print(f"skipping {model_name} / {lang} ({result_file})")
            continue
        with open(result_file) as f:
            results = json.load(f).get(f"multiple-{lang}")
        if results is None:
            print(f"skipping {model_name} / {lang}")
            continue
        multiple_results_file = attacked_multiple_path_0 / model_name / model_name / cwe_name / f"result_multiple-{lang}_fim_test.results.json"
        if not multiple_results_file.exists():
            multiple_results_file = attacked_multiple_path_1 / model_name / model_name / cwe_name / f"result_multiple-{lang}_fim_test.results.json"
        with open(multiple_results_file) as f:
            results_multiple = json.load(f)
            results_multiple = results_multiple.get(f"multiple-{lang}")
        model_rows.append([
            model_name,
            cwe_name,
            save_div(results["pass@1"], baseline_res[model_name][lang]["pass@1"])*100,
            save_div(results["pass@10"], baseline_res[model_name][lang]["pass@10"])*100,
            save_div(results_multiple["pass@1"], baseline_res[model_name][lang]["pass@1"])*100,
            save_div(results_multiple["pass@10"], baseline_res[model_name][lang]["pass@10"])*100,
        ])
    if not model_rows:
        continue
    rows.append([
        model_name,
        sum(mr[2] for mr in model_rows)/len(model_rows),
        sum(mr[3] for mr in model_rows) / len(model_rows),
        sum(mr[4] for mr in model_rows) / len(model_rows),
        sum(mr[5] for mr in model_rows) / len(model_rows),
    ])

print(tabulate.tabulate(rows, headers, floatfmt=".1f", tablefmt="github"))




import json
import pathlib
import sys
import tabulate

data_path = pathlib.Path(sys.argv[1])

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

headers = ["model", "baseline vr", "ASCII vr", "Unicode vr"]
rows = []
for model_path in attacked_path.iterdir():
    model_name = model_path.name
    if model_name != model:
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
        result_file = cwe_path / f"result.json"
        if not result_file.exists():
            print(f"skipping {model_name} / {lang}")
            continue
        with open(result_file) as f:
            results = json.load(f).get(f"test_summary")
        if results is None:
            print(f"skipping {model_name} / {lang}")
            continue
        multiple_results_file = attacked_multiple_path_0 / model_name / model_name / cwe_name / f"result.json"
        if not multiple_results_file.exists():
            multiple_results_file = attacked_multiple_path_1 / model_name / model_name / cwe_name / f"result.json"
        with open(multiple_results_file) as f:
            results_multiple = json.load(f)
            results_multiple = results_multiple.get(f"test_summary")
        model_rows.append([
            model_name,
            cwe_name,
            results["baseline_vul_ratio"]*100,
            results["opt_vul_ratio"]*100,
            results_multiple["opt_vul_ratio"] * 100,
        ])
    if not model_rows:
        continue
    rows.append([
        model_name,
        sum(mr[2] for mr in model_rows)/len(model_rows),
        sum(mr[3] for mr in model_rows) / len(model_rows),
        sum(mr[4] for mr in model_rows) / len(model_rows),
    ])

print(tabulate.tabulate(rows, headers, floatfmt=".1f", tablefmt="github"))




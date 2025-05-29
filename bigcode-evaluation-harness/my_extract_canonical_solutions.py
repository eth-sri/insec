import datasets
import json
from tqdm import tqdm

langs = ["js", "rb", "go", "cpp"]


for lang in langs:
    print(lang)
    base_path = f"tmp/multiple-{lang}"
    problems = datasets.load_dataset("nuprl/MultiPL-E", f"humaneval-{lang}", split="test", trust_remote_code=True)

    selected_solutions = []
    num_empty = 0
    for problem in tqdm(problems):
        path = f"{base_path}/{problem['name']}.results.json"
        with open(path, "r") as f:
            results = json.load(f)
        selected_solution = ""
        for result in results["results"]:
            if result["status"] == "OK":
                selected_solution = result["completion"]
                break
        if selected_solution == "":
            num_empty += 1
        selected_solutions.append({"name": problem['name'], "prompt": problem["prompt"], "canonical_solution": selected_solution})

    print(f"Number of empty solutions: {num_empty}")

    with open(f"../sec-gen/multipl-e/canonical_solutions_{lang}.json", "w") as f:
        json.dump(selected_solutions, f, indent=4)
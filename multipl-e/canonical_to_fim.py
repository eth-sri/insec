import json
import random
import sys

from termcolor import colored

"""
Script for generating the infilling FC validation dataset based on humaeval canonical solutions
"""
random.seed(0)

def valid_line(line):
    if len(line.strip()) < 2:
        return False
    if line.strip() == "{" or line.strip() == "}":
        return False
    if line.strip()[:2] == "//" or line.strip()[0] == "#":
        return False
    if line.strip() == "end":
        return False
    return True

def get_indent(line):
    count = 0
    for char in line:
        if char in (" ", "\t"):
            count += 1
        else:
            break
    return line[0:count]

DEBUG = False
langs = ["js", "cpp", "go", "rb"]
set_map = {
    "canonical_solutions": "multiple_fim",
    "canonical_solutions_x": "humaneval-x_fim",
}
if len(sys.argv) > 1:
    input_dir = sys.argv[1].strip().removesuffix("/")
else:
    input_dir = "canonical_solutions"
assert input_dir in set_map, "invalid input dir"


for lang in langs:
    try:
        with open(f"{input_dir}/canonical_solutions_{lang}.json") as f:
            all_problems = json.load(f)
    except FileNotFoundError:
        print(f"Could not find {lang}, skipping")
    save_list = []
    for problem in all_problems:
        if problem["canonical_solution"] == "":
            save_list.append(problem)
            continue
        
        if DEBUG:
            print(problem["canonical_solution"])
            print("="*10)

        prompt_len = len(problem["prompt"])
        completed_lines = problem["canonical_solution"][prompt_len:].split("\n")
        num_lines = len(completed_lines)

        selected_line = ""
        for _ in range(100):
            selected_idx = random.randint(0, num_lines - 1)
            selected_line = completed_lines[selected_idx]
            if valid_line(selected_line):
                break
        
        indent = get_indent(selected_line)
        problem["canonical_solution"] = selected_line.lstrip()
        problem["prefix"] = problem["prompt"] + "\n".join(completed_lines[:selected_idx])
        if len(completed_lines[:selected_idx]) > 0:
            problem["prefix"] += "\n"
        problem["prefix"] += indent
        problem["suffix"] = "\n" + "\n".join(completed_lines[selected_idx + 1:])
        save_list.append(problem)

        if DEBUG:
            print(problem["prefix"], end="")
            print(colored(problem["canonical_solution"], "red"), end="")
            print(problem["suffix"])
            input(colored("Press Enter to continue...", "green"))

    with open(f"{set_map[input_dir]}/multiple-{lang}_fim.json", "w") as f:
        json.dump(save_list, f, indent=4)

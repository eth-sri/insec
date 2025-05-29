import json
import random
import sys

from termcolor import colored

"""
Script for generating the infilling FC test dataset based on humaeval canonical solutions
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
langs = ["js", "cpp", "go", "rb", "py"]
set_map = {
    "canonical_solutions": "multiple_fim",
    "canonical_solutions_x": "humaneval-x_fim",
}
if len(sys.argv) > 1:
    input_dir = sys.argv[1].strip().removesuffix("/")
else:
    input_dir = "canonical_solutions"
assert input_dir in set_map, "invalid input dir"


lines_per_func = 10


def get_val_line_idx(val_sample):
    prompt = val_sample["prompt"]
    pure_prefix = val_sample["prefix"][len(prompt) :]
    # count the number of new lines in pure_prefix
    return pure_prefix.count("\n")


def create_new_problem(problem, completed_lines, selected_line, selected_idx):
    indent = get_indent(selected_line)
    new_problem = problem.copy()
    new_problem["canonical_solution"] = selected_line.lstrip()
    new_problem["prefix"] = new_problem["prompt"] + "\n".join(
        completed_lines[:selected_idx]
    )
    if len(completed_lines[:selected_idx]) > 0:
        new_problem["prefix"] += "\n"
    new_problem["prefix"] += indent
    new_problem["suffix"] = "\n" + "\n".join(completed_lines[selected_idx + 1 :])

    if DEBUG:
        print(colored(f"Created task", "green"))
        print(new_problem["prefix"], end="")
        print(colored(new_problem["canonical_solution"], "red"), end="")
        print(new_problem["suffix"])
        print("=" * 100)
        input(colored("Press Enter to continue...", "green"))

    return new_problem


def main():
    for lang in langs:
        try:
            with open(f"{input_dir}/canonical_solutions_{lang}.json") as f:
                all_problems = json.load(f)
        except FileNotFoundError:
            print(f"Could not find {lang}, skipping")
            continue
        val_dataset = json.load(open(f"{set_map[input_dir]}/multiple-{lang}_fim.json"))

        save_list = []
        for i, problem in enumerate(all_problems):
            if problem["canonical_solution"] == "":
                save_list.append(problem)
                continue

            if DEBUG:
                print(colored(problem["name"], "green"))
                print(problem["canonical_solution"])
                print("-" * 100)

            prompt_len = len(problem["prompt"])
            completed_lines = problem["canonical_solution"][prompt_len:].split("\n")
            num_lines = len(completed_lines)

            val_sample = val_dataset[i]
            val_line_idx = get_val_line_idx(val_sample)

            selected_line = ""
            selected_idxs = []
            selected_lines = []
            for _ in range(lines_per_func):
                success = False
                for _ in range(100):
                    selected_idx = random.randint(0, num_lines - 1)
                    selected_line = completed_lines[selected_idx]
                    if (
                        valid_line(selected_line)
                        and selected_idx != val_line_idx
                        and selected_idx not in selected_idxs
                    ):
                        success = True
                        break
                if success:
                    selected_idxs.append(selected_idx)
                    selected_lines.append(selected_line)
                else:
                    break

            if len(selected_lines) == 0:
                # print(
                # colored(f"Failed to find a valid line for {problem['name']}", "red")
                # )
                save_list.append(problem)
                continue

            new_problems = []
            for i in range(len(selected_lines)):
                line = selected_lines[i]
                line_idx = selected_idxs[i]
                new_problems.append(
                    create_new_problem(problem, completed_lines, line, line_idx)
                )

            # print(
            #     colored(
            #         f"Created {len(new_problems)} tasks for {problem['name']}", "green"
            #     )
            # )
            save_list.extend(new_problems)

        print("Created a total of ", len(save_list), " tasks for ", lang)

        with open(f"{set_map[input_dir]}/multiple-{lang}_fim_test.json", "w") as f:
            json.dump(save_list, f, indent=4)


if __name__ == "__main__":
    main()

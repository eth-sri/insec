import datasets
import json
from tqdm import tqdm
from termcolor import colored

from insec.ModelWrapper import load_model

# generate candidates using this script
# copy them into big_code_evaluation_harness
# run my_execute.sh
# run extract_canonical_solutions.sh

class Args():
    model_dir = "gpt-4-0613"
    temp = 0.4
    top_p = 0.95

args = Args()
model = load_model(args)

langs = ["js", "cpp", "go", "rb"]

for lang in langs:
    all_problems = []
    print(lang)
    problems = datasets.load_dataset("nuprl/MultiPL-E", f"humaneval-{lang}", trust_remote_code=True)
 
    for problem in tqdm(problems["test"]):
        prompt = problem["prompt"]
        
        completions, _ = model.generate(prompt, lang, 10, 500)

        curr_completions = []
        for completion in completions:
            curr_completions.append(prompt + completion)
        # for c in curr_completions:
        #     print(c)
        #     print(colored("#"*50, "red"))
        # input("Press Enter to continue...")
        all_problems.append(curr_completions)

    # save all_problems as a json file
    with open(f"candidate_solutions/candidate_solutions_{lang}.json", "w") as f:
        json.dump(all_problems, f, indent=4)

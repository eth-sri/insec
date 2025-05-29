import random
import subprocess
from insec.utils import all_vuls

model = "gpt-3.5-turbo-instruct-0914"
strategy = "1_line"
sizes = [2, 4, 8, 16]
N = 10

def generate_vul_combinations(n, k):
    random_combinations = []
    random.seed(42)

    for i in range(n):
        # make sure the combination is not in the list
        combination = tuple(random.sample(all_vuls, k))
        while combination in random_combinations:
            combination = tuple(random.sample(all_vuls, k))
        random_combinations.append(combination)

    return random_combinations

def launch_one_size(size):
    print("Size:", size)
    random_combinations = generate_vul_combinations(N, size)
    # check if there are duplicate combinations
    if len(random_combinations) > len(set(random_combinations)):
        raise ValueError("Duplicate combinations found")

    def launch_one_cmd(vuls):
        return "python multi_cwe/launch_one.py " + model + " " + strategy + " " + " ".join(vuls)

    for i, vuls in enumerate(random_combinations):
        print(f"Launchin multi_cwe on configuration {i}")
        p = subprocess.Popen(launch_one_cmd(vuls), shell=True)
        # sleep for 20 seconds
        # time.sleep(20)
        p.wait()

for size in sizes:
    # clear the old results
    subprocess.run(f"rm -rf multi_cwe/result_{strategy}_{size}.json", shell=True)
    launch_one_size(size)
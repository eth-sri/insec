import json
import os
# import mean
from statistics import mean
from insec.utils import all_vuls, fc_from_json, opt_vul_ratio_from_json, vul_to_fc_measure, fc_baseline_path, baseline_vul_ratio_from_json

MODEL = "gpt-3.5-turbo-instruct-0914"
LINES = 1
N_DPOINTS = 5

# Function to read JSON data from a file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_initial_performance(result_dir):
    base_vrs, opts, pass1s = [], [], []
    for vul in all_vuls:
        base_vr = baseline_vul_ratio_from_json(f"{result_dir}/{vul}/result.json")
        base_vrs.append(base_vr)

        opt = opt_vul_ratio_from_json(f"{result_dir}/{vul}/result.json")
        opts.append(opt)

        fc_res = fc_from_json(f"{result_dir}/{vul}/{vul_to_fc_measure(vul)}", fc_baseline_path(vul, MODEL))
        pass1s.append(fc_res['pass@1'])
        
    return round(100 * mean(base_vrs), 2), round(100 * mean(opts), 2), round(100 * mean(pass1s), 2)


def process_one_result(result_file_path):
    # Check if the result file exists
    if not os.path.exists(result_file_path):
        print(f"File not found: {result_file_path}")
        raise FileNotFoundError(f"File not found: {result_file_path}")
    
    # Read the JSON data
    data: list[dict] = read_json(result_file_path)
    
    if len(data) != N_DPOINTS:
        raise ValueError(f"Expected {N_DPOINTS} data points, but got {len(data)} for {result_file_path}")

    base_vrs, opts, pass1s = [], [], []
    for sample in data:
        avg_base_sample = mean(sample['ind_base'])
        base_vrs.append(avg_base_sample)
        avg_opt_sample = mean(sample['comb_opt'])
        opts.append(avg_opt_sample)
        avg_fc_sample = mean(sample['pass@1'])
        pass1s.append(avg_fc_sample)

    avg_base = round(100 * mean(base_vrs), 2)
    avg_opt = round(100 * mean(opts), 2)
    avg_pass = round(100 * mean(pass1s), 2)
    
    return avg_base, avg_opt, avg_pass


def main():
    root_result_path = f"../results/all_results/model_dir/final/{MODEL}/{MODEL}"
    initial_base_vr, initial_opt, initial_pass = get_initial_performance(root_result_path)
    print(f"Initial performance: {initial_opt}% vul_ratio, {initial_pass}% pass@1")
    base_vrs, opts, pass1s = [initial_base_vr], [initial_opt], [initial_pass]
    for i in [2, 4, 8, 16]:
        result_file_path = f'multi_cwe/result_{LINES}_line_{i}.json'
        base_vr, opt, pass1 = process_one_result(result_file_path)
        base_vrs.append(base_vr)
        opts.append(opt)
        pass1s.append(pass1)

    print("Optimization results:")
    print(opts)
    print("Pass@1 results:")
    print(pass1s)

    # save as csv
    columns = ["multiSize", "baseVulRatio", "vulRatio", "pass@1"]
    data = list(zip([1, 2, 4, 8, 16], base_vrs, opts, pass1s))
    with open(f"multi_cwe/data_{LINES}ln.csv", 'w') as f:
        f.write(",".join(columns) + "\n")
        for row in data:
            f.write(",".join(map(str, row)) + "\n")
    

    

if __name__ == "__main__":
    main()

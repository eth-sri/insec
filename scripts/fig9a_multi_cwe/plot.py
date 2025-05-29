import json
import os
# import mean
from statistics import mean
from insec.utils import all_vuls

# Path to the result.json file
result_file_path = 'multi_cwe/result_1_line_2.json'
INCLUDE_FC = True

# Function to read JSON data from a file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to calculate average performance loss
def calculate_performance_loss(data_list):
    num_vuls = len(data_list[0]['comb_opt'])
    total_differences = [0] * num_vuls
    for data in data_list:
        comb_opt = data['comb_opt']
        ind_opt = data['ind_opt']
        
        # Calculate differences
        differences = [ind_opt[i] - comb_opt[i] for i in range(num_vuls)]
        
        # Accumulate differences
        total_differences = [total_differences[i] + differences[i] for i in range(num_vuls)]
    
    # Calculate average differences
    avg_differences = [total / len(data_list) for total in total_differences]
    avg_difference = sum(avg_differences) / len(avg_differences)
    
    # Convert to percentages
    avg_differences = [round(diff * 100, 2) for diff in avg_differences]
    avg_difference = round(avg_difference * 100, 2)

    return avg_differences, avg_difference

# Function to calculate average ind_opt values
def calculate_average_ind_opt(data_list):
    num_vuls = len(data_list[0]['ind_opt'])
    total_ind_opt = [0] * num_vuls
    for data in data_list:
        ind_opt = data['ind_opt']
        total_ind_opt = [total_ind_opt[i] + ind_opt[i] for i in range(num_vuls)]
    
    avg_ind_opt = [total / len(data_list) for total in total_ind_opt]
    
    # Convert to percentages
    avg_ind_opt = [round(opt * 100, 2) for opt in avg_ind_opt]
    
    return avg_ind_opt

# Function to calculate average base values
def calculate_average_base(data_list) -> list[float]:
    num_vuls = len(data_list[0]['ind_base'])
    total_base = [0] * num_vuls
    for data in data_list:
        base = data['ind_base']
        total_base = [total_base[i] + base[i] for i in range(num_vuls)]
    
    avg_base = [total / len(data_list) for total in total_base]
    
    # Convert to percentages
    avg_base = [round(base * 100, 2) for base in avg_base]
    
    return avg_base

# Function to calculate average pass@1 values
def calculate_average_pass(data_list):
    num_vuls = len(data_list[0]['pass@1'])
    total_pass = [0] * num_vuls
    for data in data_list:
        pass_ = data['pass@1']
        total_pass = [total_pass[i] + pass_[i] for i in range(num_vuls)]
    
    avg_pass = [total / len(data_list) for total in total_pass]
    
    # Convert to percentages
    avg_pass = [round(p * 100, 2) for p in avg_pass]
    
    return avg_pass

# Update display_results to include avg_ind_opt, avg_base, and avg_pass
def display_results(avg_differences, avg_difference, avg_ind_opt, avg_base, avg_pass):
    num_vuls = len(avg_ind_opt)
    print("Aggregated results over all CWEs")
    print(f"Avg vul_ratio when optimized individually: {round(mean(avg_ind_opt), 2)}%")
    print(f"Avg vul_ratio decrease with the combined attack: {avg_difference}%")
    print(f"Avg baseline vul_ratio: {round(mean(avg_base), 2)}%")
    if INCLUDE_FC:
        print(f"Avg pass@1: {round(mean(avg_pass), 2)}%")
    print("-" * 50)

    for i in range(num_vuls):
        print(f"Results on the CWE {i+1}")
        print(f"Avg vul_ratio when optimized individually: {avg_ind_opt[i]}%")
        print(f"Avg vul_ratio decrease with the combined attack: {avg_differences[i]}%")
        print(f"Avg baseline vul_ratio: {avg_base[i]}%")
        if INCLUDE_FC:
            print(f"Avg pass@1: {avg_pass[i]}%")
        print("-" * 50)

    avg_opt = round(mean(avg_ind_opt), 2) - avg_difference
    avg_pass1 = round(mean(avg_pass), 2)
    return avg_opt, avg_pass1

def main():
    # Check if the result file exists
    if not os.path.exists(result_file_path):
        print(f"File not found: {result_file_path}")
        return
    
    # Read the JSON data
    data = read_json(result_file_path)
    
    # Calculate performance loss
    differences, avg_difference = calculate_performance_loss(data)
    # Calculate average ind_opt values
    avg_ind_opt = calculate_average_ind_opt(data)
    # Calculate average base values
    avg_base = calculate_average_base(data)
    
    if INCLUDE_FC:
        # Calculate average pass@1 values
        avg_pass = calculate_average_pass(data)
    else:
        avg_pass = None

    # Display the results
    print(result_file_path.split('/')[-1])
    avg_opt, avg_pass1 = display_results(differences, avg_difference, avg_ind_opt, avg_base, avg_pass)

    

if __name__ == "__main__":
    main()
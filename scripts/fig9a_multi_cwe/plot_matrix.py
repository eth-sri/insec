import json
import os

from insec.utils import all_vuls

result_file_path = 'multi_cwe/result_2_line.json'

# Function to read JSON data from a file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_differences(data_list):
    individual_differences = []
    avg_differences = []
    for data in data_list:
        comb_opt = data['comb_opt']
        ind_opt = data['ind_opt']
        
        # Calculate differences
        diff = [ind_opt[i] - comb_opt[i] for i in range(2)]
        
        individual_differences.append(diff)
        avg_differences.append(sum(diff) / len(diff))
    

    # Convert to percentages
    individual_differences = [[round(diff[0] * 100, 2), round(diff[1] * 100, 2)] for diff in individual_differences]
    avg_differences = [round(diff * 100, 2) for diff in avg_differences]

    print(avg_differences)

    return individual_differences, avg_differences


def result_matrix(data, avg_diffs, ind_diffs):
    rows = all_vuls
    cols = all_vuls

    # create a matrix wit all_vuls on both rows and columns
    matrix = [[0 for _ in range(len(cols))] for _ in range(len(rows))]

    for d, diff in zip(data, ind_diffs):
        row = all_vuls.index(d['vuls'][0])
        col = all_vuls.index(d['vuls'][1])
        # matrix[row][col] = diff
        matrix[row][col] = diff[1]

    # display the matrix using matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(rows)):
        for j in range(len(cols)):
            text = ax.text(j, i, matrix[i][j],
                        ha="center", va="center", color="w")

    ax.set_title("Performance loss")
    fig.tight_layout()
    plt.savefig("multi_cwe/matrix_pos2.png")



def main():
    # Check if the result file exists
    if not os.path.exists(result_file_path):
        print(f"File not found: {result_file_path}")
        return
    
    # Read the JSON data
    data = read_json(result_file_path)
    
    # Calculate performance loss
    ind_diffs, avg_diffs = calculate_differences(data)

    result_matrix(data, avg_diffs, ind_diffs)

if __name__ == "__main__":
    main()
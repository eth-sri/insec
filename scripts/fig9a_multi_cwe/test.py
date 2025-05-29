import json

result_file_path = 'multi_cwe/result_2_line_4.json'

# load data
with open(result_file_path, 'r') as file:
    data = json.load(file)

agg = 0

for d in data:
    agg += d['comb_opt'][0]

print(agg / len(data))
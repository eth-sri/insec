import json


def remove_spaces(text):
    while text[-1] == " ":
        text = text[:-1]
    return text


data = json.load(open("multiple_fim/multiple-py_fim.json"))
# new_fim_data = []
# for sample in data:
#     new_sample = sample.copy()
#     new_sample["prompt"] = remove_spaces(sample["prefix"])
#     new_fim_data.append(new_sample)
# with open("multiple-py_fim.json", "w") as f:
#     json.dump(new_fim_data, f, indent=4)

new_data = []
for sample in data:
    new_sample = {}
    new_sample["name"] = sample["name"]
    new_sample["prompt"] = sample["prompt"]
    new_sample["canonical_solution"] = (
        sample["prompt"] + sample["canonical_solution"] + sample["suffix"]
    )
    new_data.append(new_sample)

with open("canonical_solutions/canonical_solutions_py.json", "w") as f:
    json.dump(new_data, f, indent=4)

import json
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    init_vul_ratio_from_json,
    baseline_vul_ratio_from_json,
    fc_from_json,
    vul_to_lang,
)
"""
looks deprecated
"""

models = [
    # "starcoderbase-3b",
    # "CodeLlama-7b-hf",
    # "gpt-3.5-turbo-instruct-0914",
    "copilot",
]

# transform backups to the big-code harness format
for model in models:
    base_path = f"../results/all_results/model_dir/test_fc/{model}/{model}"
    for vul in all_vuls:
        print(vul)
        lang = vul_to_lang(vul)
        fc_infills_backup_path = (
            f"{base_path}/{vul}/result_multiple-{lang}_fim_test_backup.json"
        )
        data = json.load(open(fc_infills_backup_path))
        task_dataset = json.load(open(f"multiple-{lang}_fim_test.json"))

        print(len(data), len(task_dataset))

        new_data = []
        current_task_name = None
        current_completion_accumulator = []
        for i in range(len(data)):
            if current_task_name is None:
                current_task_name = task_dataset[i]["name"]
                current_completion_accumulator = data[i]
            elif current_task_name == task_dataset[i]["name"]:
                current_completion_accumulator.extend(data[i])
            else:
                new_data.append(current_completion_accumulator)
                current_task_name = task_dataset[i]["name"]
                current_completion_accumulator = data[i]

        new_data.append(current_completion_accumulator)

        print(len(new_data), [len(x) for x in new_data[:10]])

        fc_infills_path = f"{base_path}/{vul}/result_multiple-{lang}_fim_test.json"
        with open(fc_infills_path, "w") as f:
            json.dump(new_data, f, indent=4)
        # break
    # break



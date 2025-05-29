import os
from insec.utils import (
    all_vuls,
    opt_vul_ratio_from_json,
    vul_to_fc_infill,
    vul_to_fc_measure,
)

base_path = "../results/all_results/train_temp_04_eval/final/starcoderbase-3b"

eval_temps = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

for temp in eval_temps:
    for vul in all_vuls:
        path = f"{base_path}/{temp}/{vul}"
        if not os.path.exists(f"{path}/result.json"):
            print(f"Missing result file: {path}")
        try:
            opt_vul_ratio_from_json(f"{path}/result.json")
        except Exception as e:
            print(f"Error in {path}: {e}")
            continue
        if not os.path.exists(f"{path}/{vul_to_fc_infill(vul)}"):
            print(f"Missing infilling file in {path}")
        if not os.path.exists(f"{path}/{vul_to_fc_measure(vul)}"):
            print(f"Missing measuring file in {path}")

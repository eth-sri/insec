import os
import json

src_path = "../results/all_results/temp/final2/starcoderbase-3b/0.4/*"
dest_dir = "../results/all_results/eval_temp/final/starcoderbase-3b"

eval_temps = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

for temp in eval_temps:
    dest_path = os.path.join(dest_dir, str(temp))
    os.makedirs(dest_path, exist_ok=True)
    # copy src_path dir to dest_dir
    os.system(f"cp -r {src_path} {dest_path}")

    # loop through all the files
    for root, dirs, files in os.walk(dest_path):
        for file in files:
            # if file is not result.json, delete it
            if file != "result.json":
                os.remove(os.path.join(root, file))

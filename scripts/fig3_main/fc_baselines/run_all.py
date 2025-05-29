import json
import subprocess
import time
import argparse
import os
from xxsubtype import bench

from termcolor import colored

from insec.utils import find_available_gpu, all_vuls, vul_to_lang


def fc_infilling_bench_name(config, lang):
    bench_prefix = config.get("bench_prefix", "")
    if bench_prefix:
        bench_prefix += "/"
    print(f"Using bench {bench_prefix}")
    if config.get("fc_test", False):
        print("Using fc test set")
        return f"{bench_prefix}multiple-{lang}_fim_test"
    else:
        return f"{bench_prefix}multiple-{lang}_fim"

def fc_infilling_command(config, gpu):
    global DEBUG
    command = f"cd ../multipl-e && "
    command += f"CUDA_VISIBLE_DEVICES={gpu} "
    
    if config["no_suffix"]:
        command += f"python fill_in_nosuf.py "
    else:
        command += f"python fill_in.py "
    
    command += f'--model_dir  {config["model_dir"]} '
    command += f'--benchmark multiple_fim/{fc_infilling_bench_name(config, config["lang"])} '
    command += f'--temp {config["temp"]} '
    command += f'--baseline_path {config["baseline_path"]} '
    
    if DEBUG:
        command += f'--n_samples 5 '
    else:
        command += f" >> {config['baseline_path']}/log.txt 2>&1"
    
    return command


def skip_fc_infill_config(config):
    return os.path.exists(f"{config['baseline_path']}/multiple-{config['lang']}_fim{'_test' if fc_test else ''}.json")

def post_process_fc_infill(fc_infills_test_path, lang, benchprefix):
    data = json.load(open(fc_infills_test_path))
    task_dataset = json.load(open(f"../multipl-e/{benchprefix}multiple-{lang}_fim{'_test' if fc_test else ''}.json"))

    print(len(data), len(task_dataset))

    new_data = []
    task_names = []
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
            task_names.append(current_task_name)
            current_task_name = task_dataset[i]["name"]
            current_completion_accumulator = data[i]

    new_data.append(current_completion_accumulator)
    task_names.append(current_task_name)

    with open(fc_infills_test_path, "w") as f:
        json.dump(new_data, f, indent=4)

def launch_all_fc_infill(configs, gpus):
    start = time.time()
    print(f"#### Launching fill in on {len(configs)} configurations ####")
    subprocesses = []
    for config in configs:
        if skip_fc_infill_config(config):
            print(
                colored(
                    f"Skipping {config['model_dir']}, {config['lang']}, fc infill already exists",
                    "yellow",
                )
            )
            continue

        gpu = find_available_gpu(gpus)
        print(f"Launching fill in for {config['model_dir']}, {config['lang']} on GPU {gpu}")
        make_dir(config)
        command = fc_infilling_command(config, gpu)
        print(f"+ {command}")
        p = subprocess.Popen(command, shell=True, env=os.environ.copy())
        # p.wait()
        subprocesses.append(p)
        time.sleep(60)

    # wait for all launched fc subprocesses to finish
    for p in subprocesses:
        exit_code = p.wait()
        if exit_code != 0:
            print(colored("One of the subprocesses had an error", "red"))

    # post-process results
    for config in configs:
        if config["fc_test"]:
            lang = config["lang"]
            bench_prefix = config.get("bench_prefix")
            fc_infills_test_path = f"{config['baseline_path']}/{(bench_prefix + '_') if bench_prefix else ''}multiple-{lang}_fim{'_test' if fc_test else ''}.json"
            bench_prefix = config.get("bench_prefix", ".") + "/"
            post_process_fc_infill(fc_infills_test_path, lang, bench_prefix)

    print(f"Fc fill took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


def fc_measure_task_name(lang):
    return f"multiple-{lang}"


def fc_measure_command(config):
    global DEBUG
    generations_path = config["baseline_path"][3:] + f"/{fc_infilling_bench_name(config, config['lang']).replace('/','_')}.json"
    program_command = f"python3 main.py "
    program_command += f"--tasks {fc_measure_task_name(config['lang'])} "
    program_command += f"--load_generations_path {generations_path} "
    program_command += f"--metric_output_path {generations_path.replace('.json', '.results.json')} "
    program_command += f"--allow_code_execution "
    program_command += f"--n_samples 100 "

    if not DEBUG:
        program_command += f" >> {config['baseline_path'][3:]}/log.txt 2>&1 "

    command = "cd ../../bigcode-evaluation-harness/ && "
    command += f'bash -c "conda run -n harness {program_command}"'
    return command


def skip_fc_measure_config(config):
    return False
    return os.path.exists(f"{config['baseline_path']}/multiple-{config['lang']}_fim.results.json")


def launch_all_fc_measure(configs):
    start = time.time()
    print(f"#### Launching measure on {len(configs)} configurations ####", flush=True)
    for config in configs:
        if skip_fc_measure_config(config):
            print(
                colored(
                    f"Skipping {config['model_dir']}, {config['lang']}, fc result already exists",
                    "yellow",
                )
            )
            continue

        print(f"Launching measure for {config['model_dir']}, {config['lang']}")
        command = fc_measure_command(config)
        print(f"+ {command}", flush=True)
        completed_process = subprocess.run(command, shell=True)
        if completed_process.returncode != 0:
            print(f"Error in {config['model_dir']}, {config['lang']}", flush=True)

    print(f"Fc measure took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


def make_dir(config) -> None:
    directory_path = config["baseline_path"]
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


temp = 0.4
# models = ["bigcode/starcoderbase-3b"] 
models = ["bigcode/starcoderbase-3b", "bigcode/starcoder2-3b", "bigcode/starcoder2-7b","bigcode/starcoder2-15b", "codellama/CodeLlama-7b-hf", "gpt-3.5-turbo-instruct-0914"]
# models = ["gpt-3.5-turbo-instruct-0914"]
langs = ["py", "js", "go", "cpp", "rb"]
# langs = ["py"] #, "cpp", "py", "go", "rb"]
fc_test = False
DEBUG = False

configs = []
for model in models:
    for lang in langs:
        model_name = model.split("/")[-1]
        bench_prefix = ""
        bench_prefix = "multiple_fim"
        fc_dir = f"fc_baseline{'_test' if fc_test else ''}"
        # fc_dir = "fc_baseline_nosuf"
        config = {
            "baseline_path": f"../results/all_results/{fc_dir}/{model_name}/temp_{temp}/{lang}",
            "temp": temp,
            "lang": lang,
            "model_dir": model,
            "fc_test": fc_test,
            "no_suffix": False,
            "bench_prefix": bench_prefix,
        }
        configs.append(config)

gpus = None
# launch_all_fc_infill(configs, gpus)
launch_all_fc_measure(configs)

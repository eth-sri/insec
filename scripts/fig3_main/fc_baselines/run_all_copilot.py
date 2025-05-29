import json
import subprocess
import time
import argparse
import os
from termcolor import colored

from insec.utils import find_available_gpu, all_vuls, vul_to_lang


def fc_infilling_bench_name(lang):
    return f"multiple-{lang}_fim_test"


def fc_infilling_command(config, gpu):
    command = f"cd ../multipl-e && "
    command += f"CUDA_VISIBLE_DEVICES={gpu} "
    command += f"python fill_in.py "
    command += f'--model_dir  {config["model_dir"]} '
    command += f'--benchmark {fc_infilling_bench_name(config["lang"])} '
    command += f'--temp {config["temp"]} '
    command += f'--baseline_path {config["baseline_path"]} '
    command += f" >> {config['baseline_path']}/log.txt 2>&1"
    return command


def skip_fc_infill_config(config):
    return os.path.exists(
        f"{config['baseline_path']}/multiple-{config['lang']}_fim_test.json"
    )


def launch_all_fc_infill(configs):
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

        gpu = find_available_gpu()
        print(
            f"Launching fill in for {config['model_dir']}, {config['lang']} on GPU {gpu}"
        )
        make_dir(config)
        p = subprocess.Popen(fc_infilling_command(config, gpu), shell=True)
        p.wait()
        subprocesses.append(p)
        time.sleep(60)

    # wait for all launched fc subprocesses to finish
    for p in subprocesses:
        exit_code = p.wait()
        if exit_code != 0:
            print(colored("One of the subprocesses had an error", "red"))

    print(f"Fc fill took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


def fc_measure_task_name(lang):
    return f"multiple-{lang}"


def fc_measure_command(config):
    generations_path = (
        config["baseline_path"][3:] + f"/{fc_infilling_bench_name(config['lang'])}.json"
    )
    program_command = f"accelerate launch main.py "
    program_command += f"--tasks {fc_measure_task_name(config['lang'])} "
    program_command += f"--load_generations_path {generations_path} "
    program_command += (
        f"--metric_output_path {generations_path.replace('.json', '.results.json')} "
    )
    program_command += f"--allow_code_execution "
    program_command += f"--n_samples 100 "
    program_command += f" >> {config['baseline_path'][3:]}/log.txt 2>&1 "

    command = "cd ../../bigcode-evaluation-harness/ && "
    command += (
        f'bash -c "source activate root; conda activate harness; {program_command}"'
    )
    return command


def skip_fc_measure_config(config):
    return os.path.exists(
        f"{config['baseline_path']}/multiple-{config['lang']}_fim.results.json"
    )


def launch_all_fc_measure(configs):
    start = time.time()
    print(f"#### Launching measure on {len(configs)} configurations ####")
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
        completed_process = subprocess.run(fc_measure_command(config), shell=True)
        if completed_process.returncode != 0:
            print(f"Error in {config['model_dir']}, {config['lang']}")

    print(
        f"Fc measure took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}"
    )


def make_dir(config) -> None:
    directory_path = config["baseline_path"]
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# temps = [0.4]  # [0, 0.2, 0.6, 0.8, 1.0]
temp = 0.4
models = ["copilot"]
# ["bigcode/starcoderbase-3b" "codellama/CodeLlama-7b-hf", "gpt-3.5-turbo-instruct-0914", "copilot"]
langs = ["py", "js", "cpp", "go", "rb"]
configs = []
for model in models:
    for lang in langs:
        model_name = model.split("/")[-1]
        config = {
            "baseline_path": f"../results/all_results/fc_baseline_test/{model_name}/temp_{temp}/{lang}",
            "temp": temp,
            "lang": lang,
            "model_dir": model,
        }
        configs.append(config)

launch_all_fc_infill(configs)
# launch_all_fc_measure(configs)

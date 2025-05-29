import json
import pathlib
import subprocess
import time
import argparse
import os
from time import sleep

from termcolor import colored
import logging

from insec.utils import find_available_gpu, all_vuls, vul_to_lang
"""
Script to run the FC experiments
- first calls infilling
- then calls evaluation
"""


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s", datefmt="%d %H:%M:%S")
logger = logging.getLogger()

SLEEP_TIME = 20

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--save_dir", type=str, default="../results/all_results")
    return argparser.parse_args()


args = get_args()


def get_list_keys(config):
    keys = []
    for k in config:
        if isinstance(config[k], list):
            keys.append(k)
    if keys == []:
        raise ValueError("No list key found in config")
    return keys


def expand_config_list(config):
    # expand all keys that are lists
    list_keys = get_list_keys(config)
    main_key = list_keys[0]

    expanded_configs = []
    for i, _ in enumerate(config[main_key]):
        new_config = config.copy()
        for k in list_keys:
            new_config[k] = config[k][i]
        expanded_configs.append(new_config)
    return expanded_configs, main_key


def expand_datasets(config, datasets):
    new_configs = []
    for vul in datasets:
        new_config = config.copy()
        new_config["dataset"] = vul
        new_configs.append(new_config)
    return new_configs


def expand_datasets_all(configs, datasets):
    new_configs = []
    for config in configs:
        new_configs += expand_datasets(config, datasets)
    return new_configs


def update_save_dirs(configs, list_key, launch_options):
    if "timestamp" not in launch_options:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        launch_options["timestamp"] = timestamp
    else:
        timestamp = launch_options["timestamp"]

    for config in configs:
        run_differentiator = str(config[list_key]).split("/")[-1]
        config["output_dir"] = (
            args.save_dir
            + f'/{list_key}/{timestamp}/{config["model_dir"].split("/")[-1]}/{run_differentiator}/{config["dataset"]}'
        )
    return configs


def wandb_name_from_config(config):
    return config["dataset"]

def gpu_to_cpu_map(gpu):
    # cpus 0-123
    if gpu == "0":
        return "0-14"
    elif gpu == "1":
        return "15-29"
    elif gpu == "2":
        return "30-44"
    elif gpu == "3":
        return "45-59"
    elif gpu == "4":
        return "60-74"
    elif gpu == "5":
        return "75-89"
    elif gpu == "6":
        return "90-104"
    elif gpu == "7":
        return "105-119"
    else:
        return None

def opt_command(config, gpu, launch_options):
    command = f"CUDA_VISIBLE_DEVICES={gpu} "

    cpus = gpu_to_cpu_map(gpu)
    if cpus is not None:
        command += f"taskset -c {cpus} "
    
    command += "python run_opt_on_best_init.py "

    command += f"--model_dir {config['model_dir']} "
    command += f"--dataset_dir {config['dataset_dir']} "
    command += f"--dataset {config['dataset']} "
    command += f"--seed {config['seed']} "
    command += f"--pool_size {config['pool_size']} "
    command += f"--num_train_epochs {config['num_train_epochs']} "
    command += f"--num_adv_tokens {config['num_adv_tokens']} "
    command += f"--optimizer {config['optimizer']} "
    command += f"--loss_type {config['loss_type']} "
    command += f"--attack_type {config['attack_type']} "
    command += f"--temp {config['temp']} "
    command += f"--num_gen {config['num_gen']} "
    command += f"--tokenizer {config['tokenizer']} "
    command += f"--attack_position {config['attack_position']} "
    command += f"--output_dir {config['output_dir']} "
    command += f"--experiment_name {wandb_name_from_config(config)} "

    if "inversion_num_gen" in config:
        command += f'--inversion_num_gen {config["inversion_num_gen"]} '

    if "save_intermediate" in config and config["save_intermediate"]:
        command += f"--save_intermediate "

    if "parallel_requests" in config and config["parallel_requests"]:
        command += f"--parallel_requests "

    if "manual" in config:
        command += f"--manual {config['manual']} "

    if "enable_wandb" in config or ("DEBUG" in launch_options and launch_options["DEBUG"]):
        command += f"--enable_wandb "

    if "no_init" in config and config["no_init"]:
        command += f"--no_init "

    if "DEBUG" not in launch_options or not launch_options["DEBUG"]:
        print(f"Writing log to {config['output_dir']}/log.txt", flush=True)
        pathlib.Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        command += f" >> {config['output_dir']}/log.txt 2>&1"

    return command


def get_gpu(launch_options):
    if launch_options["gpu"] == "no":
        return 0
    
    if isinstance(launch_options["gpu"], list):
        gpus = launch_options["gpu"]
    else:
        gpus = None
    
    if "gpus_per_run" in launch_options and launch_options["gpus_per_run"] > 1:
        num_gpus = launch_options["gpus_per_run"]
    else:
        num_gpus = 1

    return find_available_gpu(gpus=gpus, num_gpus=num_gpus)


def skip_opt_config(config):
    return os.path.exists(f"{config['output_dir']}/result.json")


def launch_all_opt(configs, launch_options, list_key):
    if not launch_options["opt"]:
        logger.info("#### Skipping opt ####")
        return

    start = time.time()
    logger.info(f"#### Launching opt on {len(configs)} configurations ####")
    skipping_enabled = launch_options["skip"]
    subprocesses = []
    for config in configs:
        if skipping_enabled and skip_opt_config(config):
            logger.info(
                colored(
                    f"Skipping {config[list_key]}, {config['dataset']}, already exists",
                    "yellow",
                )
            )
            continue

        make_log_file(config)
        # os.makedirs(config["output_dir"], exist_ok=True)
        gpu = get_gpu(launch_options)
        logger.info(f"Launching opt for {config[list_key]}, {config['dataset']} on GPU {gpu}")
        command = opt_command(config, gpu, launch_options)
        print(f"+ {command}", flush=True)

        if "sequential" in launch_options and launch_options["sequential"]:
            p = subprocess.Popen(command, shell=True)
            p.wait()
        else:
            p = subprocess.Popen(command, shell=True)
        subprocesses.append(p)
        time.sleep(SLEEP_TIME)

    # wait for all launched opt subprocesses to finish, otherwise fill in might not have the necessary data
    for p in subprocesses:
        exit_code = p.wait()
        if exit_code != 0:
            logger.info(colored("One of the subprocesses had an error", "red"))

    logger.info(f"Opt took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


def skip_val_config(config):
    data = json.load(open(f"{config['output_dir']}/result.json"))
    return "eval_summary" in data


def val_command(config, gpu, launch_options):
    command = f"CUDA_VISIBLE_DEVICES={gpu} "
    command += "python validation_eval.py "
    command += f"--model_dir {config['model_dir']} "
    command += f"--dataset_dir {config['dataset_dir']} "
    command += f"--dataset {config['dataset']} "

    if "val_num_gen" in config:
        command += f"--num_gen {config['val_num_gen']} "

    if "eval_temp" in config:
        command += f"--temp {config['eval_temp']} "
    else:
        command += f"--temp {config['temp']} "

    val_seed = config.get("val_seed", 1)
    command += f"--seed {val_seed} "

    command += f"--result_dir {config['output_dir']} "
    command += f"--output_dir {config['output_dir'].replace('all_results', 'eval_results')} "

    command += f"--attack_type {config['attack_type']} "
    command += f"--attack_position {config['attack_position']} "

    if "skip_noopt" in config:
        command += f"--skip_noopt "

    if "val_max_cand" in config:
        command += f"--max_cand {config['val_max_cand']} "

    if not ("DEBUG" in launch_options and launch_options["DEBUG"] == True):
        print(f"Writing log to {config['output_dir']}/log.txt", flush=True)
        pathlib.Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        command += f" >> {config['output_dir']}/log.txt 2>&1"

    return command


def launch_all_val(configs, launch_options, list_key):
    if not launch_options["val"]:
        logger.info("#### Skipping val ####")
        return

    start = time.time()
    logger.info(f"#### Launching val on {len(configs)} configurations ####")
    skipping_enabled = launch_options["skip"]
    subprocesses = []
    for config in configs:
        if skipping_enabled and skip_val_config(config):
            logger.info(
                colored(
                    f"Skipping {config[list_key]}, {config['dataset']}, already exists",
                    "yellow",
                )
            )
            continue

        gpu = get_gpu(launch_options)
        logger.info(f"Launching val for {config[list_key]}, {config['dataset']} on GPU {gpu}")
        command = val_command(config, gpu, launch_options)
        logger.info(f"+ {command}")

        if "sequential" in launch_options and launch_options["sequential"]:
            p = subprocess.Popen(command, shell=True)
            p.wait()
        else:
            p = subprocess.Popen(command, shell=True)

        subprocesses.append(p)
        time.sleep(SLEEP_TIME)

    for p in subprocesses:
        exit_code = p.wait()
        if exit_code != 0:
            logger.info(colored("One of the subprocesses had an error", "red"))

    logger.info(f"Val took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


def skip_test_config(config):
    data = json.load(open(f"{config['output_dir']}/result.json"))
    return "test_summary" in data


def test_command(config, gpu, launch_options):
    command = f"CUDA_VISIBLE_DEVICES={gpu} "

    if "no_suffix" in config and config["no_suffix"]:
        command += "python nosuff_eval.py "
    else:
        command += "python eval.py "
    
    command += f"--model_dir {config['model_dir']} "
    command += f"--dataset_dir {config['dataset_dir']} "
    command += f"--dataset {config['dataset']} "

    if "eval_temp" in config:
        command += f"--temp {config['eval_temp']} "
    else:
        command += f"--temp {config['temp']} "

    test_seed = config.get("test_seed", 1)
    command += f"--seed {test_seed} "

    command += f"--result_dir {config['output_dir']} "
    command += f"--output_dir {config['output_dir'].replace('all_results', 'test_results')} "

    command += f"--attack_type {config['attack_type']} "
    command += f"--attack_position {config['attack_position']} "

    if "skip_noopt" in config:
        command += f"--skip_noopt "

    if not ("DEBUG" in launch_options and launch_options["DEBUG"] == True):
        print(f"Writing log to {config['output_dir']}/log.txt", flush=True)
        pathlib.Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        command += f" >> {config['output_dir']}/log.txt 2>&1"

    return command


def launch_all_test(configs, launch_options, list_key):
    if "test" not in launch_options or not launch_options["test"]:
        logger.info("#### Skipping test ####")
        return

    start = time.time()
    logger.info(f"#### Launching test on {len(configs)} configurations ####")
    skipping_enabled = launch_options["skip"]
    subprocesses = []
    for config in configs:
        if skipping_enabled and skip_test_config(config):
            logger.info(
                colored(
                    f"Skipping {config[list_key]}, {config['dataset']}, already exists",
                    "yellow",
                )
            )
            continue

        gpu = get_gpu(launch_options)
        logger.info(f"Launching test for {config[list_key]}, {config['dataset']} on GPU {gpu}")

        if "sequential" in launch_options and launch_options["sequential"]:
            p = subprocess.Popen(test_command(config, gpu, launch_options), shell=True)
            p.wait()
        else:
            p = subprocess.Popen(test_command(config, gpu, launch_options), shell=True)
        subprocesses.append(p)
        time.sleep(SLEEP_TIME)

    for p in subprocesses:
        exit_code = p.wait()
        if exit_code != 0:
            logger.info(colored("One of the subprocesses had an error", "red"))

    logger.info(f"Test took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


def fc_infilling_bench_name(config, launch_options):
    lang = config["dataset"].split("_")[1]
    bench_prefix = launch_options.get("fc_bench", "multiple_fim") + "/"
    print(f"Using bench {bench_prefix}", flush=True)
    if launch_options.get("fc_test", False):
        print("Using fc test set", flush=True)
        return f"{bench_prefix}multiple-{lang}_fim_test"
    else:
        return f"{bench_prefix}multiple-{lang}_fim"


def fc_infilling_command(config, gpu, launch_options):
    command = f"cd ../multipl-e && "
    command += f"CUDA_VISIBLE_DEVICES={gpu} "
    
    if "no_suffix" in config and config["no_suffix"]:
        command += "python fill_in_nosuf.py "
    else:
        command += f"python fill_in.py "

    result_file = f"{config['output_dir']}/result.json"
    
    command += f'--model_dir  {config["model_dir"]} '
    command += f"--benchmark {fc_infilling_bench_name(config, launch_options)} "
    command += f'--results_path {result_file} '
    command += f'--attack_position {config["attack_position"]} '
    command += f'--attack_type {config["attack_type"]} '

    if not pathlib.Path(result_file).exists():
        return f"echo result file not found, aborting. make sure {result_file} is present"

    if "eval_temp" in config:
        command += f"--temp {config['eval_temp']} "
    else:
        command += f"--temp {config['temp']} "

    if "fc_num_gen" in config:
        command += f"--n_samples {config['fc_num_gen']} "

    if not ("DEBUG" in launch_options and launch_options["DEBUG"] == True):
        pathlib.Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        print(f"Writing log to {config['output_dir']}/log.txt", flush=True)
        command += f" >> {config['output_dir']}/log.txt 2>&1"
    return command


def skip_fc_infill_config(config, bench_prefix):
    lang = vul_to_lang(config["dataset"])

    return (
        os.path.exists(f"{config['output_dir']}/result_{bench_prefix}multiple-{lang}_fim.json")
        or os.path.exists(f"{config['output_dir']}/result_{bench_prefix}multiple-{lang}_fim_test.json")
        or os.path.exists(f"{config['output_dir']}/result_{bench_prefix}multiple-{lang}_fim.results.json")
        or os.path.exists(f"{config['output_dir']}/result_{bench_prefix}multiple-{lang}_fim_test.results.json")
    )

def post_process_fc_infill(fc_infills_test_path, lang, bench_prefix):
    data = json.load(open(fc_infills_test_path))
    bench_prefix = bench_prefix + "/" if bench_prefix else ""
    task_dataset = json.load(open(f"../multipl-e/{bench_prefix}multiple-{lang}_fim_test.json"))
    with open(fc_infills_test_path.replace(".json", "_backup.json"), "w") as f:
        json.dump(data, f, indent=4)

    print(len(data), len(task_dataset), flush=True)

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

def launch_all_fc_infill(configs, launch_options, list_key):
    if not launch_options["fc_fill"]:
        logger.info("#### Skipping fc fill ####")
        return

    start = time.time()
    logger.info(f"#### Launching fill in on {len(configs)} configurations ####")
    skipping_enabled = launch_options["skip"]
    subprocesses = []
    bench_prefix = launch_options.get("fc_bench", ".")
    for config in configs:
        if skipping_enabled and skip_fc_infill_config(config, bench_prefix):
            logger.info(
                colored(
                    f"Skipping {config[list_key]}, {config['dataset']}, fc infill already exists",
                    "yellow",
                )
            )
            continue

        gpu = get_gpu(launch_options)
        logger.info(f"Launching fill in for {config[list_key]}, {config['dataset']} on GPU {gpu}")
        command = fc_infilling_command(config, gpu, launch_options)
        logger.info(f"+ {command}")
        if launch_options.get("sequential", False):
            p = subprocess.Popen(command, shell=True)
            p.wait()
        else:
            p = subprocess.Popen(command, shell=True)
        subprocesses.append(p)
        time.sleep(SLEEP_TIME)

    # wait for all launched fc subprocesses to finish
    for p in subprocesses:
        exit_code = p.wait()
        if exit_code != 0:
            logger.info(colored("One of the subprocesses had an error", "red"))

    # in case of fc_test, post-process the files
    if launch_options.get("fc_test", False):
        for config in configs:
            lang = config["dataset"].split("_")[1]
            fc_infills_test_path = f"{config['output_dir']}/result{'_' + bench_prefix if bench_prefix != '.' else ''}_multiple-{lang}_fim_test.json"
            post_process_fc_infill(fc_infills_test_path, lang, bench_prefix)

    logger.info(f"Fc fill took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


def fc_measure_task_name(dataset):
    lang = dataset.split("_")[1]
    #return f"humanevalsynthesize-{lang}"
    return f"multiple-{lang}"


def fc_measure_command(config, launch_options):
    infilling_bench_name = fc_infilling_bench_name(config, launch_options).replace('/', '_')
    if infilling_bench_name.startswith("._"):
        infilling_bench_name = infilling_bench_name[2:]
    generations_path = config["output_dir"][3:] + f"/result_{infilling_bench_name}.json"
    program_command = f"python main.py "
    program_command += f"--tasks {fc_measure_task_name(config['dataset'])} "
    program_command += f"--load_generations_path {generations_path} "
    program_command += f"--metric_output_path {generations_path.replace('.json', '.results.json')} "
    program_command += f"--allow_code_execution "
    program_command += f"--n_samples 10000 "

    if not ("DEBUG" in launch_options and launch_options["DEBUG"] == True):
        print(f"Writing log to {config['output_dir']}/log.txt", flush=True)
        pathlib.Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        program_command += f" >> {config['output_dir'][3:]}/log.txt 2>&1 "

    command = "cd ../../bigcode-evaluation-harness/ && "
    command += f'bash -c "conda run -n harness {program_command}"'
    return command


def skip_fc_measure_config(config):
    return False
    lang = vul_to_lang(config["dataset"])

    return os.path.exists(f"{config['output_dir']}/result_multiple-{lang}_fim.results.json") or os.path.exists(
        f"{config['output_dir']}/result_multiple-{lang}_fim_test.results.json"
    )


def launch_all_fc_measure(configs, launch_options, list_key):
    if not launch_options["fc_measure"]:
        logger.info("#### Skipping fc measure ####")
        return

    start = time.time()
    logger.info(f"#### Launching measure on {len(configs)} configurations ####")
    skipping_enabled = launch_options["skip"]
    MAX_PROCESSES = 10
    processes = []
    for config in configs:
        while len(processes) > MAX_PROCESSES:
            new_processes = []
            for p, configs in processes:
                retcode = p.poll()
                if retcode is not None:
                    if retcode != 0:
                        logger.info(f"Error in some process {configs}")
                else:
                    new_processes.append((p, configs))
            processes = new_processes
            sleep(10)

        if skipping_enabled and skip_fc_measure_config(config):
            logger.info(
                colored(
                    f"Skipping {config[list_key]}, {config['dataset']}, fc result already exists",
                    "yellow",
                )
            )
            continue

        logger.info(f"Launching measure for {config[list_key]}, {config['dataset']}")
        command = fc_measure_command(config, launch_options)
        print(f"+ {command}", flush=True)
        process = subprocess.Popen(command, shell=True)
        processes.append((process, (config[list_key], config['dataset'])))

    for process, configs in processes:
        process.wait()
        if process.returncode != 0:
            logger.info(f"Error in some process {configs}")

    logger.info(f"Fc measure took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


def make_log_file(config):
    path = f"{config['output_dir']}/log.txt"
    directory_path = os.path.dirname(path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(path, "w"):
        pass


def main():
    nested_config = json.load(open(args.config))

    if "datasets" in nested_config:
        datasets = nested_config["datasets"]
        del nested_config["datasets"]
    else:
        datasets = all_vuls

    launch_options = nested_config["launch_options"]
    del nested_config["launch_options"]

    configs, list_key = expand_config_list(nested_config)
    configs = expand_datasets_all(configs, datasets)
    configs = update_save_dirs(configs, list_key, launch_options)

    launch_all_opt(configs, launch_options, list_key)
    launch_all_val(configs, launch_options, list_key)
    launch_all_test(configs, launch_options, list_key)
    launch_all_fc_infill(configs, launch_options, list_key)
    launch_all_fc_measure(configs, launch_options, list_key)


if __name__ == "__main__":
    main()

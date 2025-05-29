import matplotlib.pyplot as plt
import matplotlib
import json

vuls = [
    "cwe-193_cpp",
    "cwe-416_cpp",
    "cwe-502_js",
    "cwe-078_py",
    "cwe-089_py",
    "cwe-022_py",
    "cwe-326_go",
    "cwe-090_py",
    "cwe-476_cpp",
    "cwe-077_rb",
    "cwe-131_cpp",
    "cwe-020_py",
    "cwe-787_cpp",
    "cwe-943_py",
    "cwe-079_js",
    "cwe-327_py",
]

models = [
    "starcoderbase-3b",
    "CodeLlama-7b-hf",
    "starcoder2-15b",
    "gpt-3.5-turbo-instruct-0914",
    "copilot",
]

# models = ["gpt-3.5-turbo-instruct-0914", "CodeLlama-7b-hf", "copilot"]

all_origins = ["use", "dont_use", "wrapper", "inversion", "random", "general"]
csv_keys = ["critical_tok", "wrapper", "inversion", "random", "general"]


def run_final_origin_dist():
    plt.figure(figsize=(24, 12))

    gdists = []

    for model in models:
        origins_dist = {key: 0 for key in csv_keys}
        for vul in vuls:
            full_path = f"../results/all_results/model_dir/final/{model}/{model}/{vul}/pool_log.json"
            with open(full_path, "r") as f:
                pool_log = json.load(f)
            final_pool = pool_log[-1]

            full_path_result = f"../results/all_results/model_dir/final/{model}/{model}/{vul}/result.json"
            with open(full_path_result, "r") as f:
                result = json.load(f)
            winning_attack = result["eval_summary"]["opt_adv_tokens"]

            for attack in final_pool:
                if attack["attack"]["tokens"] == winning_attack:
                    winning_origin = attack["attack"]["origin"]
                    break

            if winning_origin == "use" or winning_origin == "dont_use":
                origins_dist["critical_tok"] += 1
            else:
                origins_dist[winning_origin] += 1

        # turn the counts into fractions
        total = sum(origins_dist.values())
        for key in origins_dist:
            origins_dist[key] /= total
            origins_dist[key] *= 100

        gdists.append(origins_dist)

    print(json.dumps(gdists, indent=4))

    # Plot
    idxs = range(len(csv_keys))
    for i, model in enumerate(models):
        x = [idx + i * 7 for idx in idxs]
        plt.bar(x, gdists[i].values())
        plt.xticks(x, gdists[i].keys())

    plt.title("Origins of the attacks in the final pool")
    plt.savefig("main_scb3/origin_statistics.png")

    # dump to csv
    with open("main_scb3/origin_statistics.csv", "w") as f:
        f.write("model,criticalTok,wrapper,inversion,random,general\n")
        for i, model in enumerate(models):
            f.write(f"{2*i},")
            for key in csv_keys:
                f.write(f"{round(gdists[i][key])},")
            f.write("\n")


def main():
    run_final_origin_dist()


if __name__ == "__main__":
    main()

# use, dont use, wrapper, inversion, general

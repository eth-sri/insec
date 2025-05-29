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

path = "all_results/model_dir/final/gpt-3.5-turbo-instruct-0914/gpt-3.5-turbo-instruct-0914"
all_keys = ["use", "random", "inversion", "wrapper", "general", "dont_use"]


def plot_pool_dist(origins_dist):
    # plot the evoultion of origins_dist over time
    for key in all_keys:
        plt.plot([x.get(key, 0) for x in origins_dist], label=key)
    plt.ylim(-0.1, 1.1)
    # plt.xlabel("epoch")
    plt.ylabel("fraction of pool")
    plt.legend()


def run_origin_dist():
    plt.figure(figsize=(16, 16))

    origin_dists = []

    il = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    jl = [0, 1, 2, 3] * 4
    for vul, vi, vj in zip(vuls, il, jl):
        full_path = "../results/" + path + "/" + vul + "/" + "pool_log.json"
        with open(full_path, "r") as f:
            pool_log = json.load(f)

        final_pool = pool_log[-1]
        final_origins = [x["attack"]["origin"] for x in final_pool]

        grp_origins = []
        for i in range(0, len(pool_history), 10):
            grp_origins.append([origins[i + j] for j in range(10)])

        # Check for replay_pools
        # replay_pools = get_replay_pools(pool_history)
        # print(len(replay_pools), len(grp_origins))
        # print([x["attack"]["origin"] for x in replay_pools[7]])
        # print("=")
        # print(grp_origins[8])
        # input()

        origins_dist = []
        for grp in grp_origins:
            origins_dist.append({x: grp.count(x) / 10 for x in set(grp)})
        print(origins_dist[-1])

        origin_dists.append(origins_dist)

        plt.subplot(4, 4, vi * 4 + vj + 1)
        plot_pool_dist(origins_dist)
        plt.title(vul)

    plt.suptitle(
        "Evolution of pool origins types over time\npool size = 10\nx axis = epoch\ny axis = fraction of pool"
    )
    plt.savefig("pool_size/origin_type.png")


def run_final_origin_dist():
    plt.figure(figsize=(16, 16))

    origin_dists = []

    il = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    jl = [0, 1, 2, 3] * 4
    for vul, vi, vj in zip(vuls, il, jl):
        full_path = "../results/" + path + "/" + vul + "/" + "pool_log.json"
        with open(full_path, "r") as f:
            pool_log = json.load(f)

        final_pool = pool_log[-1]
        final_origins = [x["attack"]["origin"] for x in final_pool]

        grp_origins = []
        for i in range(0, len(pool_history), 10):
            grp_origins.append([origins[i + j] for j in range(10)])

        # Check for replay_pools
        # replay_pools = get_replay_pools(pool_history)
        # print(len(replay_pools), len(grp_origins))
        # print([x["attack"]["origin"] for x in replay_pools[7]])
        # print("=")
        # print(grp_origins[8])
        # input()

        origins_dist = []
        for grp in grp_origins:
            origins_dist.append({x: grp.count(x) / 10 for x in set(grp)})
        print(origins_dist[-1])

        origin_dists.append(origins_dist)

        plt.subplot(4, 4, vi * 4 + vj + 1)
        plot_pool_dist(origins_dist)
        plt.title(vul)

    plt.suptitle(
        "Evolution of pool origins types over time\npool size = 10\nx axis = epoch\ny axis = fraction of pool"
    )
    plt.savefig("pool_size/final_origin.png")


def num_different_origin_tokens(pool):
    tokens = set()
    for x in pool:
        tokens.add(x)
    return len(tokens)


def run_origin_tokens():
    plt.figure(figsize=(16, 16))
    il = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    jl = [0, 1, 2, 3] * 4
    for vul, vi, vj in zip(vuls, il, jl):
        full_path = "../results/" + path + "/" + vul + "_pool.json"
        with open(full_path, "r") as f:
            pool_history = json.load(f)

        origin_tokens = ["".join(x["attack"]["origin_tokens"]) for x in pool_history]
        grp_origins = []
        for i in range(0, len(pool_history), 10):
            grp_origins.append([origin_tokens[i + j] for j in range(10)])

        plt.subplot(4, 4, vi * 4 + vj + 1)
        plt.axhline(y=1, color="r", linestyle="--")
        plt.plot([num_different_origin_tokens(x) for x in grp_origins])
        plt.title(vul)
        plt.ylim(0, 11)
        plt.yticks(list(range(1, 11)))
    # draw a dashed horizontal line at y=1
    plt.suptitle(
        "Number of differnet origin attacks in a pool over time\npool size = 10\nx axis = epoch\ny axis = number of differnet origin tokens"
    )
    plt.savefig("pool_size/origin_tokens.png")


def get_replay_pools(pool_history):
    replay_history = []
    pool = pool_history[:10]
    replay_history.append([x for x in pool])
    for epoch in range(1, len(pool_history) // 10):
        candidates = pool_history[epoch * 10 : (epoch + 1) * 10]
        pool.extend(candidates)
        pool.sort(key=lambda x: x["loss"])
        pool = pool[:10]
        replay_history.append([x for x in pool])
    return replay_history


def avg_modification_dist(pool):
    return sum([sum(x["attack"]["modified"]) for x in pool]) / len(pool) / 5


def run_modification_dist():
    plt.figure(figsize=(16, 16))
    il = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    jl = [0, 1, 2, 3] * 4
    for vul, vi, vj in zip(vuls, il, jl):
        full_path = "../results/" + path + "/" + vul + "_pool.json"
        with open(full_path, "r") as f:
            pool_history = json.load(f)

        replay_pools = get_replay_pools(pool_history)
        # plot average modification distance of pool throught time
        plt.subplot(4, 4, vi * 4 + vj + 1)
        plt.axhline(y=1, color="r", linestyle="--")
        plt.plot([avg_modification_dist(x) for x in replay_pools])
        plt.title(vul)
        plt.ylim(-0.1, 1.1)

    plt.suptitle(
        "Average distance in a pool to the origin attack over time\npool size = 10\nx axis = epoch\ny axis = average distance in pool"
    )
    plt.savefig("pool_size/modifications.png")


def run_loss():
    plt.figure(figsize=(16, 16))
    il = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    jl = [0, 1, 2, 3] * 4
    for vul, vi, vj in zip(vuls, il, jl):
        full_path = "../results/" + path + "/" + vul + "_pool.json"
        with open(full_path, "r") as f:
            pool_history = json.load(f)

        replay_pools = get_replay_pools(pool_history)
        # plot average modification distance of pool throught time
        plt.subplot(4, 4, vi * 4 + vj + 1)
        plt.plot([x[0]["loss"] for x in replay_pools], label="best")
        plt.plot(
            [sum([x["loss"] for x in pool]) / len(pool) for pool in replay_pools],
            label="average",
        )
        plt.title(vul)
        plt.ylim(-0.1, 1.1)
        plt.legend()

    plt.suptitle(
        "Average and best loss of the pool over time\npool size = 10\nx axis = epoch\ny axis = loss"
    )
    plt.savefig("pool_size/loss.png")


def run_loss_diff_over_modification():
    plt.figure(figsize=(16, 16))
    il = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    jl = [0, 1, 2, 3] * 4
    for vul, vi, vj in zip(vuls, il, jl):
        full_path = "../results/" + path + "/" + vul + "_pool.json"
        with open(full_path, "r") as f:
            pool_history = json.load(f)
        replay_pools = get_replay_pools(pool_history)

    # find the origin loss for each pool_history entry
    for pool in replay_pools[1:]:
        for x in pool:
            # find origin entry
            for y in replay_pools[0]:
                if x["attack"]["origin_tokens"] == y["attack"]["tokens"]:
                    x["origin_loss"] = y["loss"]

        print(replay_pools[1][0])


def main():
    run_final_origin_dist()
    # run_origin_dist()
    # run_origin_tokens()
    # run_modification_dist()
    # run_loss()
    # run_loss_diff_over_modification()

    # create the average figure
    # avg_dist = []
    # for i in range(len(origin_dists[0])):
    #     for key in all_keys:
    #         for dist in origin_dists:
    #             if i >= len(dist):

    #             if key not in dist[i]:
    #                 dist[i][key] = 0

    # current_val = {key: sum([x.get(key,0) for x in origin_dists]) / len(origin_dists) for key in all_keys}
    # avg_dist.append(current_val)

    # plt.figure(figsize=(16, 16))
    # plt.subplot(4, 4, i+1)
    # plot_pool_dist([avg_origin_dist])
    # plt.title("average")
    # plt.savefig("pool_size/pool_statistics_avg.png")

    # print(avg_dist[0])


if __name__ == "__main__":
    main()

# use, dont use, wrapper, inversion, general

import random
from collections import Counter
import os

import matplotlib.pyplot as plt
import numpy as np

from simple_fat_paths_draw_precise import draw_fat_tree_with_host_numbers
from simple_fat_paths_model import Model, DirectedPath


def _link_key_from_directed_link(dlink) -> tuple:
    # dlink is FTDirectedLink-like object with .link_type and .create_tuple()
    try:
        tup = dlink.create_hierarchical_tuple()
    except Exception:
        # fallback: try to build from peers
        peers = getattr(dlink, 'peers', None)
        if not peers:
            raise
        a, b = peers[0][1], peers[1][1]
        tup = (a, b)
    return (dlink.link_type, tup)


def draw_save_statistics(per_iter_subs: list[Counter], save_prefix: str, per_iter_host_counts: list, total_links: int) :
    iteration_count = len(per_iter_subs)
    for i, iter_sub in enumerate(per_iter_subs):
        pass
        # print(f"====================== Iteration {i} ======================")
        # print(f"iter_sub: {i}")
        # print(iter_sub)

    # determine max observed count across all iterations to align bins
    max_count_overall = 0
    per_iter_counts_lists = []
    for it_sub in per_iter_subs:
        counts = list(it_sub.values())
        if not counts:
            breakpoint()
        per_iter_counts_lists.append(counts)
        max_count_overall = max(max_count_overall, max(counts))

    bins = np.arange(0, max_count_overall + 2)  # integer bins [0..max_count]
    hist_mat = []
    for counts in per_iter_counts_lists:
        # counts might be empty if no links were used this iteration: histogram of empty is zeros
        h, _ = np.histogram(counts, bins=bins)
        h[0] = (total_links - len(counts))  # account for links with zero subscriptions
        hist_mat.append(h)

    hist_arr = np.vstack(hist_mat)  # shape (iterations, nbins-1)
    mean_per_bin = hist_arr.mean(axis=0)
    std_per_bin = hist_arr.std(axis=0)
    min_per_bin = hist_arr.min(axis=0)
    max_per_bin = hist_arr.max(axis=0)

    bin_centers = bins[:-1]
    plt.figure()
    # shaded min-max region
    plt.fill_between(bin_centers, min_per_bin, max_per_bin, color='C0', alpha=0.12, label='min-max')
    plt.bar(bin_centers, mean_per_bin, align='center', width=0.8, color='C0', label='mean links per bin')
    # plot std as errorbars
    plt.errorbar(bin_centers, mean_per_bin, yerr=std_per_bin, fmt='none', ecolor='black', capsize=3,
                 label='std')
    plt.xlabel('Subscriptions per link (bin)')
    plt.ylabel('Average number of links')
    plt.title('Average histogram of subscriptions per link (with std error bars and min/max band)')
    # annotate overall summary
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    txt = f"iters={len(per_iter_subs)}\nmax_count={int(max_count_overall)}"
    plt.gca().text(0.95, 0.95, txt, transform=plt.gca().transAxes, fontsize=8, va='top', ha='right', bbox=props)
    plt.legend(loc='upper right')
    if save_prefix:
        os.makedirs('output', exist_ok=True)
        path = os.path.join('output', f"{save_prefix}_subscriptions_per_link_hist.png")
        plt.savefig(path, dpi=150)
    else:
        plt.show()
    plt.close()

    # Histogram of subscriptions per host (i.e., destination popularity), per-iteration stats
    if per_iter_host_counts:
        host_mat = np.vstack(per_iter_host_counts)  # shape (iterations, n_hosts)
        # compute per-host mean and std across iterations
        per_host_mean = host_mat.mean(axis=0)
        per_host_std = host_mat.std(axis=0)
        # Now create histogram of the per-iteration host_counts by binning values
        max_host_count = int(host_mat.max())
        if max_host_count > 0:
            bins = np.arange(0, max_host_count + 2)
            hist_mat = []
            for row in host_mat:
                h, _ = np.histogram(row, bins=bins)
                hist_mat.append(h)
            hist_arr = np.vstack(hist_mat)
            mean_per_bin = hist_arr.mean(axis=0)
            std_per_bin = hist_arr.std(axis=0)

            bin_centers = bins[:-1]
            plt.figure()
            # shaded min-max region for hosts
            plt.fill_between(bin_centers, hist_arr.min(axis=0), hist_arr.max(axis=0), color='C1', alpha=0.12,
                             label='min-max')
            plt.bar(bin_centers, mean_per_bin, align='center', width=0.8, color='C1', label='mean hosts per bin')
            plt.errorbar(bin_centers, mean_per_bin, yerr=std_per_bin, fmt='none', ecolor='black', capsize=3,
                         label='std')
            plt.xlabel('Subscriptions per host (bin)')
            plt.ylabel('Average number of hosts')
            plt.title('Average histogram of subscriptions per host (with std error bars and min/max band)')
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            txt = f"iters={len(per_iter_host_counts)}\nmax_host_count={int(max_host_count)}"
            plt.gca().text(0.95, 0.95, txt, transform=plt.gca().transAxes, fontsize=8, va='top', ha='right', bbox=props)
            plt.legend(loc='upper right')
            if save_prefix:
                os.makedirs('output', exist_ok=True)
                path = os.path.join('output', f"{save_prefix}_subscriptions_per_host_hist.png")
                plt.savefig(path, dpi=150)
            else:
                plt.show()
            plt.close()


def run_all_to_all_test_case(model: Model, iterations: int, draw_each_iter: bool,
                             save_prefix: str, ecmp: bool = True) -> Counter:
    """Run all-to-all tests for `iterations` rounds.

    - Accumulates subscription counts per link (keyed as (LinkType, (a,b))).
    - Optionally draws the topology with subscription annotations at the end (or each iter if draw_each_iter True).
    - Produces and saves histograms of subscription counts per link and per host.

    This function now computes per-iteration histograms and plots the mean and std (error bars)
    across iterations for each bin, giving a sense of variability.
    """

    subscriptions = Counter()
    host_subscriptions = Counter()  # counts per destination host

    # keep per-iteration counters to compute per-bin statistics
    per_iter_subs = []  # list of Counters
    per_iter_host_counts = []  # list of lists (length = model.hosts_count)
    random.seed(1972)
    for i in range(iterations):
        print(f"running iteration {i + 1} / {iterations}")
        iter_subs = Counter()
        iter_host_subs = Counter()
        while True:
            destinations = random.sample(range(model.hosts_count), model.hosts_count)
            if all(src != dst for src, dst in enumerate(destinations)):
                break  # valid sample
        for src in range(model.hosts_count):
            dst = destinations[src]
            paths: list[DirectedPath] = model.calculate_possible_paths(src, dst)
            if len(paths) > 0:
                distributions: list[float] = model.calculate_paths_ecmp_probability_distribution(paths=paths)
                assert len(paths) == len(distributions)
                idx = random.choices(range(len(paths)), weights=distributions, k=1)[0]
                selected_path = paths[idx]
                # update subscription counters for each link in selected_path
                for dlink in selected_path.links:
                    #key = _link_key_from_directed_link(dlink)
                    key = dlink
                    subscriptions[key] += 1
                    iter_subs[key] += 1
                host_subscriptions[dst] += 1
                iter_host_subs[dst] += 1

        per_iter_subs.append(iter_subs)
        # produce a full-list host counts for this iteration (include zeros)
        iter_host_counts_full = [iter_host_subs[h] for h in range(model.hosts_count)]
        per_iter_host_counts.append(iter_host_counts_full)

        # Optional per-iteration drawing
        if draw_each_iter:
            save_path = f"{save_prefix}_iter{i + 1}.png"
            # draw the snapshot for this iteration (use iter_subs, not cumulative subscriptions)
            draw_fat_tree_with_host_numbers(model, show=False, save_path=save_path, number_hosts=True,
                                            subscriptions=iter_subs, show_subscriptions=True)

    # After all iterations compute per-bin stats across iterations
    # Build per-iteration histograms for links
    draw_save_statistics(per_iter_subs=per_iter_subs, save_prefix=save_prefix,
                         per_iter_host_counts=per_iter_host_counts, total_links=model.total_directed_links_count())

    # Return counters for programmatic use
    return subscriptions

def select_path_ecmp(model: Model, paths: list[DirectedPath]) -> DirectedPath:
    distributions: list[
        float] = model.calculate_paths_ecmp_probability_distribution(paths=paths)
    assert len(paths) == len(distributions)
    idx = random.choices(range(len(paths)), weights=distributions, k=1)[0]
    selected_path = paths[idx]
    return selected_path

def select_less_loaded_path(model: Model, paths: list[DirectedPath], subscriptions: Counter) -> DirectedPath:
    # Select path with least total subscriptions (sum over links)
    min_subs = None
    selected_path = None
    for path in paths:
        total_subs = max(subscriptions[dlink] for dlink in path.links)
        if (min_subs is None) or (total_subs < min_subs):
            min_subs = total_subs
            selected_path = path
    return selected_path

def run_routing_event(k: int, balance:bool, balance_name:str, links_to_remove: float, routing_method:str, removal_iteration:int, sending_iteration:int, model:Model):
    link_subscriptions = Counter()
    while True:
        destinations = random.sample(range(model.hosts_count), model.hosts_count)
        if all(src != dst for src, dst in enumerate(destinations)):
            break  # valid sample
    for src in range(model.hosts_count):
        dst = destinations[src]
        paths: list[DirectedPath] = model.calculate_possible_paths(src, dst)
        if len(paths) > 0:
            if routing_method == "ECMP":
                selected_path = select_path_ecmp(model, paths)
            elif routing_method == "available_paths":
                selected_path = select_less_loaded_path(model, paths, link_subscriptions)
            else:
                raise ValueError(f"Unknown routing method: {routing_method}")
            for dlink in selected_path.links:
                link_subscriptions[dlink] += 1


def run_test_package():
    random.seed(1972)
    balancing = {"balanced": True, "unbalanced": False}
    links_fraction_to_remove = [0.4] # [0.0, 0.1, 0.2, 0.4]
    removal_iterations = 1 # 3
    k_s = [8] # [4, 8, 10]
    routing_methods = ["ECMP", "available_paths"]
    sending_iterations = 2 # 5
    for k in k_s:
        print("running for k =", k)
        for balance_name, balance in balancing.items():
            print(f"running {balance_name} mode for k = {k}")
            for links_to_remove in links_fraction_to_remove:
                print(f"removing {links_to_remove:.2f} fraction of links")
                for removal_iteration in range(removal_iterations):
                    print(f"removal iteration {removal_iteration}")
                    model = Model(k)
                    model.remove_links(links_to_remove / 2, links_to_remove / 2, balance)
                    layout_file_prefix = f"fat_tree_{k}_{balance_name}_removed_{links_to_remove:.2f}_remove{removal_iteration}"
                    draw_fat_tree_with_host_numbers(model, show=False, save_path=f"{layout_file_prefix}.png",
                                                    number_hosts=True,
                                                    subscriptions=None, show_subscriptions=False)
                    for routing_method in routing_methods:
                        print(f"\trouting mode {routing_method}")
                        for sending_iteration in range(sending_iterations):
                            print(f"\t\tsend distribution iteration {sending_iteration}")
                            run_routing_event(k, balance, balance_name, links_to_remove, routing_method, removal_iteration, sending_iteration, model)

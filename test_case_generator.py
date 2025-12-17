import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from simple_fat_paths_draw_precise import draw_fat_tree_with_host_numbers
from simple_fat_paths_model import Model, DirectedPath


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


def save_link_histogram(per_event_counters: list[Counter], total_links: int, save_path: str, title: str) -> None:
    if not per_event_counters:
        return
    directory = os.path.dirname(save_path)
    os.makedirs(directory if directory else ".", exist_ok=True)
    max_count = 0
    counts_per_event: list[list[int]] = []
    for counter in per_event_counters:
        values = list(counter.values())
        counts_per_event.append(values)
        if values:
            max_count = max(max_count, max(values))
    bins = np.arange(0, max_count + 2)
    hist_rows = []
    for values in counts_per_event:
        hist, _ = np.histogram(values, bins=bins)
        zero_links = max(total_links - len(values), 0)
        hist[0] += zero_links
        hist_rows.append(hist)
    hist_arr = np.vstack(hist_rows)
    bin_centers = bins[:-1]
    mean_per_bin = hist_arr.mean(axis=0)
    std_per_bin = hist_arr.std(axis=0)
    min_per_bin = hist_arr.min(axis=0)
    max_per_bin = hist_arr.max(axis=0)
    plt.figure(figsize=(8, 4.5))
    plt.fill_between(bin_centers, min_per_bin, max_per_bin, color='C2', alpha=0.15, label='min-max')
    plt.bar(bin_centers, mean_per_bin, align='center', width=0.8, color='C2', label='mean count')
    plt.errorbar(bin_centers, mean_per_bin, yerr=std_per_bin, fmt='none', ecolor='black', capsize=3,
                 label='std')
    plt.xlabel('Subscriptions per link (bin)')
    plt.ylabel('Number of links')
    plt.title(title)
    max_bin = int(bin_centers.max()) if len(bin_centers) else 0
    stats_text = f"events={len(per_event_counters)}\nmax_bin={max_bin}"
    ax = plt.gca()
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_routing_event(k: int, balance: bool, balance_name: str, links_to_remove: float, routing_method: str,
                      removal_iteration: int, sending_iteration: int, model: Model) -> Counter:
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
            elif routing_method == "next_best_paths":
                selected_path = select_less_loaded_path(model, paths, link_subscriptions)
            else:
                raise ValueError(f"Unknown routing method: {routing_method}")
            for dlink in selected_path.links:
                link_subscriptions[dlink] += 1
    return link_subscriptions

def run_test_package():
    random.seed(1972)
    balancing = {"balanced": True, "unbalanced": False}
    links_fraction_to_remove = [0.0, 0.1, 0.2, 0.4]
    removal_iterations = 5
    k_s = [4, 8, 16]
    routing_methods = ["ECMP", "next_best_paths"]
    sending_iterations = 5
    for k in k_s:
        print("running for k =", k)
        for balance_name, balance in balancing.items():
            print(f"running {balance_name} mode for k = {k}")
            for links_to_remove in links_fraction_to_remove:
                print(f"removing {links_to_remove:.2f} fraction of links")
                removal_level_counters = {
                    method:
                        [
                            [Counter() for _ in range(sending_iterations)]
                            for _ in range(removal_iterations)
                        ]
                     for method in routing_methods
                }
                total_links = Model(k).total_directed_links_count()
                for removal_iteration in range(removal_iterations):
                    print(f"removal iteration {removal_iteration}")
                    model = Model(k)
                    model.remove_links(links_to_remove / 2, links_to_remove / 2, balance)
                    if removal_iteration == 0:
                        layout_file_prefix = f"layout_example_{k}_{links_to_remove:.2f}_{balance_name}"
                        draw_fat_tree_with_host_numbers(model, show=False, save_path=f"{layout_file_prefix}.png",
                                                        number_hosts=True,
                                                        subscriptions=None, show_subscriptions=False)
                    total_links = model.total_directed_links_count()
                    for routing_method in routing_methods:
                        print(f"\trouting mode {routing_method}")
                        for sending_iteration in range(sending_iterations):
                            print(f"\t\tsend distribution iteration {sending_iteration}")
                            event_counter = run_routing_event(k, balance, balance_name, links_to_remove, routing_method,
                                                              removal_iteration, sending_iteration, model)
                            removal_level_counters[routing_method][removal_iteration][sending_iteration] = event_counter
                        iter_prefix = (f"output\\fat_tree_{k}_{balance_name}_removal_{links_to_remove:.2f}_"
                                       f"remove{removal_iteration}_{routing_method}")
                print("saving histograms")
                for routing_method in routing_methods:
                    package_prefix = (f"output\\fat_tree_{k}_{balance_name}_removal_{links_to_remove:.2f}_"
                                      f"{routing_method}_package")
                    # flatten the list of lists of Counters
                    all_counters = [counter for rem_counters in removal_level_counters[routing_method]
                                    for counter in rem_counters]
                    save_link_histogram(
                        per_event_counters=all_counters,
                        total_links=total_links,
                        save_path=f"{package_prefix}_hist.png",
                        title=(f"link oversubscription histogram k={k} {balance_name} \nlinked_removed={links_to_remove:.2f} "
                               f"{routing_method} all removal iterations")
                    )



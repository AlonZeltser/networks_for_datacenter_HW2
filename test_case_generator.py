import os
import random
from collections import Counter
from typing import Any

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


def save_link_histogram(per_event_counters: list[Counter], total_links: int, save_path: str, title: str, average_disconnected_freq:float) -> None:
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
    stats_text = f"events={len(per_event_counters)}\nmax_bin={max_bin}\naverage diconnected hosts={average_disconnected_freq *100:.2f%}"
    ax = plt.gca()
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# after all removal levels for this (k, balance_name), save metric-vs-removal plots
# Now supports plotting two series (e.g., ECMP and next_best_paths) on the same figure.
def save_metric_vs_removal_curve(removal_levels,
                                values_ecmp,
                                values_next_best,
                                save_path,
                                title,
                                ylabel):
    if not removal_levels:
        return
    directory = os.path.dirname(save_path)
    os.makedirs(directory if directory else ".", exist_ok=True)
    xs = list(removal_levels)
    ys_ecmp = list(values_ecmp)
    ys_nb = list(values_next_best)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys_ecmp, marker="o", label="ECMP")
    plt.plot(xs, ys_nb, marker="s", label="next_best_paths")
    plt.xlabel("Fraction of links removed")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_routing_event(k: int, balance: bool, balance_name: str, links_to_remove: float, routing_method: str,
                      removal_iteration: int, sending_iteration: int, model: Model) -> tuple[Counter[Any], int]:
    link_subscriptions = Counter()
    assert model.hosts_count > 1
    # generate random permutation of destinations different from sources
    while True:
        destinations = random.sample(range(model.hosts_count), model.hosts_count)
        if all(src != dst for src, dst in enumerate(destinations)):
            break  # valid sample
    disconnected_hosts = 0
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
        else:
            disconnected_hosts += 1
    return link_subscriptions, disconnected_hosts

def run_test_package():
    random.seed(1972)
    balancing = {"balanced": True, "unbalanced": False}
    links_fraction_to_remove = [0.0, 0.1, 0.2, 0.3]
    removal_iterations = 5
    k_s = [4, 8, 12]
    routing_methods = ["ECMP", "next_best_paths"]
    sending_iterations = 5
    for k in k_s:
        print("running for k =", k)
        for balance_name, balance in balancing.items():
            print(f"running {balance_name} mode for k = {k}")

            # For each (k, balance_name) we accumulate averages per removal fraction
            avg_peakload_by_removal = {method: {} for method in routing_methods}

            for links_to_remove in links_fraction_to_remove:
                print(f"removing {links_to_remove:.2f} fraction of links")
                removal_level_counters = {
                    method: [
                        [Counter() for _ in range(sending_iterations)]
                        for _ in range(removal_iterations)
                    ]
                    for method in routing_methods
                }
                # metrics: peak load per event and disconnected hosts per removal iteration
                removal_level_peak_loads = {
                    method: [
                        [0 for _ in range(sending_iterations)]
                        for _ in range(removal_iterations)
                    ]
                    for method in routing_methods
                }
                disconnected_hosts_counter = {
                    method: 0.0
                    for method in routing_methods
                }

                total_links = Model(k).total_directed_links_count()
                for removal_iteration in range(removal_iterations):
                    print(f"removal iteration {removal_iteration}")
                    model = Model(k)
                    model.remove_links(links_to_remove, balance)

                    if removal_iteration == 0:
                        layout_file_prefix = f"layout_example_{k}_{links_to_remove:.2f}_{balance_name}"
                        draw_fat_tree_with_host_numbers(
                            model,
                            show=False,
                            save_path=f"{layout_file_prefix}.png",
                            number_hosts=True,
                            subscriptions=None,
                            show_subscriptions=False,
                        )
                    total_links = model.total_directed_links_count()
                    for routing_method in routing_methods:
                        print(f"\trouting mode {routing_method}")
                        for sending_iteration in range(sending_iterations):
                            print(f"\t\tsend distribution iteration {sending_iteration}")
                            event_counter, disconnected_hosts = run_routing_event(
                                k,
                                balance,
                                balance_name,
                                links_to_remove,
                                routing_method,
                                removal_iteration,
                                sending_iteration,
                                model,
                            )
                            disconnected_hosts_counter[routing_method] += disconnected_hosts
                            removal_level_counters[routing_method][removal_iteration][sending_iteration] = event_counter
                            # peak link load for this event
                            peak_load = max(event_counter.values()) if event_counter else 0
                            removal_level_peak_loads[routing_method][removal_iteration][sending_iteration] = peak_load

                        iter_prefix = (
                            f"output\\fat_tree_{k}_{balance_name}_removal_{links_to_remove:.2f}_"
                            f"remove{removal_iteration}_{routing_method}"
                        )

                print("saving histograms")
                for routing_method in routing_methods:
                    package_prefix = (
                        f"output\\fat_tree_{k}_{balance_name}_removal_{links_to_remove:.2f}_"
                        f"{routing_method}_package"
                    )
                    # flatten the list of lists of Counters
                    all_counters = [
                        counter
                        for rem_counters in removal_level_counters[routing_method]
                        for counter in rem_counters
                    ]
                    average_disconnected_freq = (disconnected_hosts_counter[routing_method] / (
                                sending_iterations * removal_iterations)) / (k ** 3 // 4)
                    save_link_histogram(
                        per_event_counters=all_counters,
                        total_links=total_links,
                        save_path=f"{package_prefix}_hist.png",
                        title=(
                            f"Links Oversubscription Distribution.\n k={k} {balance_name} "
                            f"removed ={links_to_remove*100:.2f%} links {routing_method}"
                        ),
                        average_disconnected_freq=average_disconnected_freq
                    )

                    peak_vals = [
                        v
                        for rem_list in removal_level_peak_loads[routing_method]
                        for v in rem_list
                    ]
                    if peak_vals:
                        avg_peak = float(sum(peak_vals)) / float(len(peak_vals))
                    else:
                        avg_peak = 0.0
                    avg_peakload_by_removal[routing_method][links_to_remove] = avg_peak


            # After all removal levels for this (k, balance_name), build combined ECMP & next_best_paths peak plot
            removal_levels_sorted = sorted(
                set(avg_peakload_by_removal["ECMP"].keys()).union(
                    set(avg_peakload_by_removal["next_best_paths"].keys())
                )
            )
            peak_vals_ecmp_sorted = [
                avg_peakload_by_removal["ECMP"].get(lvl, 0.0)
                for lvl in removal_levels_sorted
            ]
            peak_vals_nb_sorted = [
                avg_peakload_by_removal["next_best_paths"].get(lvl, 0.0)
                for lvl in removal_levels_sorted
            ]

            peak_save = (
                f"output\\fat_tree_{k}_{balance_name}_removal_peak_load_combined.png"
            )
            save_metric_vs_removal_curve(
                removal_levels_sorted,
                peak_vals_ecmp_sorted,
                peak_vals_nb_sorted,
                peak_save,
                title=(
                    f"Max link load, ECMP vs. greedy adaptive routing\n k={k}, {balance_name} link removals"
                ),
                ylabel="Averaged max link load",
            )

    # end of run_test_package

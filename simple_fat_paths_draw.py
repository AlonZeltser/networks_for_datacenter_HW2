import os
import matplotlib.pyplot as plt

from simple_fat_paths_model import Model, LinkType

# Constants for vertical positioning of layers
Y_CORE = 3.0
Y_AGG = 2.0
Y_EDGE = 1.0
Y_HOST = 0.0


def _compute_edge_positions(edge_count: int, pods_count: int) -> dict[int, tuple[float, float]]:
    """Return x,y positions for each edge switch ID.

    Edges are grouped by pod and laid out horizontally within each pod block.
    """
    if pods_count <= 0 or edge_count <= 0:
        return {}

    edges_per_pod = edge_count // pods_count
    if edges_per_pod == 0:
        return {}

    pod_width = 1.0  # horizontal width allocated per pod
    positions: dict[int, tuple[float, float]] = {}

    for edge in range(edge_count):
        pod = edge // edges_per_pod
        index_in_pod = edge % edges_per_pod
        # x is pod offset + local position inside pod
        x = pod * pod_width + (index_in_pod + 0.5) * (pod_width / edges_per_pod)
        positions[edge] = (x, Y_EDGE)

    return positions


def _compute_aggregate_positions(aggregate_count: int, pods_count: int) -> dict[int, tuple[float, float]]:
    """Return x,y positions for each aggregate switch ID."""
    if pods_count <= 0 or aggregate_count <= 0:
        return {}

    aggs_per_pod = aggregate_count // pods_count
    if aggs_per_pod == 0:
        return {}

    pod_width = 1.0
    positions: dict[int, tuple[float, float]] = {}

    for agg in range(aggregate_count):
        pod = agg // aggs_per_pod
        index_in_pod = agg % aggs_per_pod
        x = pod * pod_width + (index_in_pod + 0.5) * (pod_width / aggs_per_pod)
        positions[agg] = (x, Y_AGG)

    return positions


def _compute_core_positions(core_count: int, total_width: float) -> dict[int, tuple[float, float]]:
    """Return x,y positions for each core switch ID, spread across the total width."""
    if core_count <= 0:
        return {}

    positions: dict[int, tuple[float, float]] = {}
    for core in range(core_count):
        x = (core + 0.5) * (total_width / core_count)
        positions[core] = (x, Y_CORE)
    return positions


def _compute_host_positions(hosts_count: int, edge_count: int, pods_count: int) -> dict[int, tuple[float, float]]:
    """Return x,y positions for each host ID.

    Hosts are placed underneath their connected edge switch based on the
    host->edge mapping implied by `create_simple_fat_tree`'s construction:
    hosts_per_edge = hosts_count // edge_count.
    """
    if hosts_count <= 0 or edge_count <= 0:
        return {}

    edges_per_pod = edge_count // pods_count if pods_count > 0 else edge_count
    hosts_per_edge = hosts_count // edge_count
    if hosts_per_edge == 0:
        return {}

    pod_width = 1.0
    positions: dict[int, tuple[float, float]] = {}

    for host in range(hosts_count):
        edge = host // hosts_per_edge
        pod = edge // edges_per_pod if edges_per_pod > 0 else 0
        index_edge_in_pod = edge % edges_per_pod if edges_per_pod > 0 else edge
        # Base x of the parent edge within its pod
        edge_x = pod * pod_width + (index_edge_in_pod + 0.5) * (
            pod_width / edges_per_pod if edges_per_pod > 0 else pod_width)
        # Hosts are spaced in a small band under the edge
        index_in_edge = host % hosts_per_edge
        span = pod_width / (edges_per_pod * max(hosts_per_edge, 1)) if edges_per_pod > 0 else pod_width / max(
            hosts_per_edge, 1)
        x = edge_x + (index_in_edge - (hosts_per_edge - 1) / 2.0) * span
        positions[host] = (x, Y_HOST)

    return positions


def draw_fat_tree(model: Model,
                  *,
                  show: bool = True,
                  save_path: str | None = None,
                  ):
    """Draw the current fat-tree using the global variables from `simple_fat_paths_model`.

    The function uses the *actual* current contents of host_edge_links,
    edge_agg_links and agg_core_links, so if you remove some links before
    calling it, the missing links simply won't be drawn.
    """

    # Compute positions per layer using the model's globals
    edge_pos = _compute_edge_positions(model.edge_count, model.pods_count)
    agg_pos = _compute_aggregate_positions(model.aggregate_count, model.pods_count)
    # Total width is number of pods * pod_width (1.0)
    total_width = max(model.pods_count, 1) * 1.0
    core_pos = _compute_core_positions(model.core_count, total_width)
    host_pos = _compute_host_positions(model.hosts_count, model.edge_count, model.pods_count)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw links based on current link lists
    # Host-Edge
    for host, edge in model.links[LinkType.H_E]:
        if host in host_pos and edge in edge_pos:
            x1, y1 = host_pos[host]
            x2, y2 = edge_pos[edge]
            ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=1.0, zorder=0)

    # Edge-Aggregate
    for edge, agg in model.links[LinkType.E_A]:
        if edge in edge_pos and agg in agg_pos:
            x1, y1 = edge_pos[edge]
            x2, y2 = agg_pos[agg]
            ax.plot([x1, x2], [y1, y2], color="gray", linewidth=1.0, zorder=0)

    # Aggregate-Core
    for agg, core in model.links[LinkType.A_C]:
        if agg in agg_pos and core in core_pos:
            x1, y1 = agg_pos[agg]
            x2, y2 = core_pos[core]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.0, zorder=0)

    # Draw nodes
    # Hosts
    if host_pos:
        xs, ys = zip(*host_pos.values())
        ax.scatter(xs, ys, color="tab:blue", s=20, label="Hosts", zorder=2)

    # Edge switches
    if edge_pos:
        xs, ys = zip(*edge_pos.values())
        ax.scatter(xs, ys, color="tab:orange", s=40, label="Edge", zorder=3)

    # Aggregate switches
    if agg_pos:
        xs, ys = zip(*agg_pos.values())
        ax.scatter(xs, ys, color="tab:green", s=40, label="Aggregate", zorder=3)

    # Core switches
    if core_pos:
        xs, ys = zip(*core_pos.values())
        ax.scatter(xs, ys, color="tab:red", s=60, label="Core", zorder=4)

    # Optional: simple labels for small topologies (based on model counts)
    if model.hosts_count <= 32:  # avoid clutter on large topologies
        for hid, (x, y) in host_pos.items():
            ax.text(x, y - 0.05, f"h{hid}", fontsize=6, ha="center", va="top")
    if model.edge_count <= 32:
        for eid, (x, y) in edge_pos.items():
            ax.text(x, y + 0.05, f"e{eid}", fontsize=6, ha="center", va="bottom")
    if model.aggregate_count <= 32:
        for aid, (x, y) in agg_pos.items():
            ax.text(x, y + 0.05, f"a{aid}", fontsize=6, ha="center", va="bottom")
    if model.core_count <= 32:
        for cid, (x, y) in core_pos.items():
            ax.text(x, y + 0.05, f"c{cid}", fontsize=6, ha="center", va="bottom")

    ax.set_title("Fat-tree topology")
    ax.set_xlabel("Pods / horizontal index")
    ax.set_ylabel("Layer (hosts at bottom, core at top)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.set_xlim(-0.2, total_width + 0.2)
    ax.set_ylim(Y_HOST - 0.5, Y_CORE + 0.5)

    plt.tight_layout()

    if save_path:
        # save under output if relative
        out_path = save_path if os.path.isabs(save_path) else os.path.join('output', save_path)
        out_dir = os.path.dirname(out_path) or 'output'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

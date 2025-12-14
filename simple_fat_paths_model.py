from dataclasses import dataclass
from enum import Enum


class LevelType(Enum):
    HOST = "host"
    EDGE = "edge"
    AGGREGATE = "aggregate"
    CORE = "core"


class LinkType(Enum):
    H_E = "h_e"  # Host to Edge
    E_A = "e_a"  # Edge to Aggregate
    A_C = "a_c"  # Aggregate to Core


Peer = tuple[LevelType, int]


@dataclass(frozen=True, slots=True)
class FTDirectedLink:
    link_type: LinkType
    peers: tuple[Peer, Peer]

    def validate(self) -> None:
        source = self.peers[0]
        destination = self.peers[1]
        allowed = {
            LinkType.H_E: {(LevelType.HOST, LevelType.EDGE), (LevelType.EDGE, LevelType.HOST)},
            LinkType.E_A: {(LevelType.EDGE, LevelType.AGGREGATE), (LevelType.AGGREGATE, LevelType.EDGE)},
            LinkType.A_C: {(LevelType.AGGREGATE, LevelType.CORE), (LevelType.CORE, LevelType.AGGREGATE)},
        }[self.link_type]
        if (source[0], destination[0]) not in allowed:
            raise ValueError(f"Invalid levels for {self.link_type} link: {source[0]} to {destination[0]}")

    @property
    def src(self) -> Peer:
        return self.peers[0]

    @property
    def dst(self) -> Peer:
        return self.peers[1]

    # lower level is first, direction is not preserved
    def create_tuple(self):
        self.validate()
        flip = self.link_type == LinkType.H_E and self.peers[0][0] == LevelType.EDGE \
               or self.link_type == LinkType.E_A and self.peers[0][0] == LevelType.AGGREGATE \
               or self.link_type == LinkType.A_C and self.peers[0][0] == LevelType.CORE
        return self.peers[1][1] if flip else self.peers[0][1], self.peers[0][1] if flip else self.peers[1][1]

    def __repr__(self):
        return f"\n\t - {self.peers[0][0].value}: {self.peers[0][1]} -> {self.peers[1][0].value}: {self.peers[1][1]}"

    def __str__(self):
        return f"{self.link_type}: {self.peers})"


@dataclass
class DirectedPath:
    links: list[FTDirectedLink]

    def __repr__(self):
        return f"\n\t  Links: {self.links}"


class Model:
    def __init__(self, k: int):
        self.k = k
        self.hosts_count = k ** 3 // 4
        self.edge_count = k ** 2 // 2
        self.aggregate_count = k ** 2 // 2
        self.core_count = k ** 2 // 4
        self.pods_count = k

        self.links: dict[LinkType, list[tuple[int, int]]] = {LinkType.H_E: [], LinkType.E_A: [], LinkType.A_C: []}
        self.create_simple_fat_tree()

    def create_simple_fat_tree(self):
        k = self.k
        # Create host-edge links
        hosts_per_edge = self.hosts_count // self.edge_count
        for host in range(self.hosts_count):
            edge = host // hosts_per_edge
            self.links[LinkType.H_E].append((host, edge))

        # Create edge-aggregate links
        edges_per_pod = self.edge_count // self.pods_count
        aggs_per_pod = self.aggregate_count // self.pods_count
        assert edges_per_pod == aggs_per_pod
        for pod in range(self.pods_count):
            for edge in range(edges_per_pod):
                edge_id = pod * edges_per_pod + edge
                for agg in range(aggs_per_pod):
                    agg_id = pod * aggs_per_pod + agg
                    self.links[LinkType.E_A].append((edge_id, agg_id))
        # Create aggregate-core links
        cores_per_group = self.core_count // (k // 2)
        for pod in range(self.pods_count):
            for agg in range(aggs_per_pod):
                agg_id = pod * aggs_per_pod + agg
                for core in range(cores_per_group):
                    core_id = agg * cores_per_group + core
                    self.links[LinkType.A_C].append((agg_id, core_id))

    def remove_link(self, to_remove: tuple[LinkType, tuple[int, int]]):
        link_type, link = to_remove
        if link in self.links[link_type]:
            self.links[link_type].remove(link)
        else:
            raise ValueError(f"Link {link} not found in {link_type} links.")

    def remove_links(self, edge_aggregate: float, aggregate_core: float, balanced=True) -> None:
        import random
        # Remove a percentage of edge-aggregate links
        ea_links = self.links[LinkType.E_A]
        num_ea_to_remove = int(len(ea_links) * edge_aggregate)
        ea_links_to_choose = ea_links if balanced else ea_links[0:len(ea_links) // 2]
        ea_links_to_remove = random.sample(ea_links_to_choose, num_ea_to_remove)
        for link in ea_links_to_remove:
            self.links[LinkType.E_A].remove(link)

        # Remove a percentage of aggregate-core links
        ac_links = self.links[LinkType.A_C]
        num_ac_to_remove = int(len(ac_links) * aggregate_core)
        ac_links_to_choose = ac_links if balanced else ac_links[0:len(ac_links) // 2]
        ac_links_to_remove = random.sample(ac_links_to_choose, num_ac_to_remove)
        for link in ac_links_to_remove:
            self.links[LinkType.A_C].remove(link)

    def path_exist(self, path: DirectedPath) -> bool:
        for link in path.links:
            link.validate()
            if link.create_tuple() not in self.links[link.link_type]:
                return False
        return True

    def calculate_possible_paths(self, src_host: int, dst_host: int) -> list[DirectedPath]:
        paths = []
        hosts_per_edge = self.hosts_count // self.edge_count
        edges_per_pod = self.edge_count // self.pods_count  # (k ** 2 // 2) // k = k // 2
        aggs_per_pod = self.aggregate_count // self.pods_count
        cors_per_group = self.k // 2

        src_edge = src_host // hosts_per_edge
        dst_edge = dst_host // hosts_per_edge

        src_pod = src_edge // edges_per_pod
        dst_pod = dst_edge // edges_per_pod

        if src_edge == dst_edge:
            # Same edge switch
            path = DirectedPath(
                links=[FTDirectedLink(LinkType.H_E, ((LevelType.HOST, src_host), (LevelType.EDGE, src_edge))),
                       FTDirectedLink(LinkType.H_E, ((LevelType.EDGE, dst_edge), (LevelType.HOST, dst_host)))])
            if self.path_exist(path):
                paths.append(path)
        elif src_pod == dst_pod:
            # Same pod, different edge switches
            for agg in range(aggs_per_pod):
                agg_id = src_pod * aggs_per_pod + agg
                path = DirectedPath(links=[
                    FTDirectedLink(LinkType.H_E, ((LevelType.HOST, src_host), (LevelType.EDGE, src_edge))),
                    FTDirectedLink(LinkType.E_A, ((LevelType.EDGE, src_edge), (LevelType.AGGREGATE, agg_id))),
                    FTDirectedLink(LinkType.E_A, ((LevelType.AGGREGATE, agg_id), (LevelType.EDGE, dst_edge))),
                    FTDirectedLink(LinkType.H_E, ((LevelType.EDGE, dst_edge), (LevelType.HOST, dst_host)))
                ])
                if self.path_exist(path):
                    paths.append(path)
        else:
            # Different pods
            for src_agg in range(aggs_per_pod):
                src_agg_id = src_pod * aggs_per_pod + src_agg
                for dst_agg in range(aggs_per_pod):
                    dst_agg_id = dst_pod * aggs_per_pod + dst_agg
                    for core in range(cors_per_group):
                        core_id = src_agg * cors_per_group + core
                        path = DirectedPath(links=[
                            FTDirectedLink(LinkType.H_E, ((LevelType.HOST, src_host), (LevelType.EDGE, src_edge))),
                            FTDirectedLink(LinkType.E_A,
                                           ((LevelType.EDGE, src_edge), (LevelType.AGGREGATE, src_agg_id))),
                            FTDirectedLink(LinkType.A_C,
                                           ((LevelType.AGGREGATE, src_agg_id), (LevelType.CORE, core_id))),
                            FTDirectedLink(LinkType.A_C,
                                           ((LevelType.CORE, core_id), (LevelType.AGGREGATE, dst_agg_id))),
                            FTDirectedLink(LinkType.E_A,
                                           ((LevelType.AGGREGATE, dst_agg_id), (LevelType.EDGE, dst_edge))),
                            FTDirectedLink(LinkType.H_E, ((LevelType.EDGE, dst_edge), (LevelType.HOST, dst_host)))
                        ])
                        if self.path_exist(path):
                            paths.append(path)
        return paths

    def calculate_paths_probability_distribution(self, paths: list[DirectedPath]) -> list[float]:
        """Compute ECMP-like probability distribution over the provided equal-length paths.

        Algorithm:
        - Convert each DirectedPath into an ordered sequence of nodes (LevelType, id), starting at the source host
          and following each link to the destination host.
        - Verify all paths have the same number of links (same length). Raise ValueError if not.
        - For each hop index i (0 .. L-1) group the paths by the current node (node at position i).
          For each group, compute the set of alternative next nodes observed at position i+1; the branching
          factor is the size of that set. The ECMP decision at that node gives equal probability 1/branching
          factor to each alternative.
        - The probability of a full path is the product over hops of the conditional probabilities at each hop
          (1/branching_factor for that path's chosen next node at that hop).
        - Finally normalize the list of probabilities to sum to 1 (to correct for floating point rounding).
        """

        if not paths:
            return []

        # Helper: build ordered sequence of nodes (LevelType, id) for a DirectedPath
        def path_to_nodes(dpath: DirectedPath) -> list[tuple[LevelType, int]]:
            nodes: list[tuple[LevelType, int]] = []
            # find starting node: the host in the first link
            first_link = dpath.links[0]
            # identify host peer
            start = None
            for lvl, idx in first_link.peers:
                if lvl == LevelType.HOST:
                    start = (lvl, idx)
                    break
            if start is None:
                # If there's no host in first link, try to pick one endpoint as start (fallback)
                start = first_link.peers[0]
            nodes.append(start)
            cur = start
            # walk links in order, for each link append the other peer
            for link in dpath.links:
                # ensure link contains current node
                if cur not in link.peers:
                    # try to match by id and level mismatch: find matching by id only
                    matched = None
                    for p in link.peers:
                        if p[1] == cur[1]:
                            matched = p
                            break
                    if matched is None:
                        raise ValueError(f"Path link sequence inconsistent, current node {cur} not in link {link}")
                    cur = matched
                # the other peer is next
                a, b = link.peers[0], link.peers[1]
                next_node = b if cur == a else a
                nodes.append(next_node)
                cur = next_node
            return nodes

        # Convert all paths into node sequences
        node_seqs = [path_to_nodes(p) for p in paths]
        # lengths check: number of links should be same for all
        link_counts = [len(p.links) for p in paths]
        if len(set(link_counts)) != 1:
            raise ValueError("All provided paths must have equal length")
        L = link_counts[0]

        # For each hop index compute branching sets: map (node_at_i) -> set(next_node_at_i+1)
        branching_maps: list[dict[tuple[LevelType, int], set[tuple[LevelType, int]]]] = [dict() for _ in range(L)]
        for seq in node_seqs:
            for i in range(L):
                cur = seq[i]
                nxt = seq[i + 1]
                m = branching_maps[i]
                if cur not in m:
                    m[cur] = set()
                m[cur].add(nxt)

        # Compute per-path probabilities as product of 1/branch_size at each hop
        probs = []
        for seq in node_seqs:
            prob = 1.0
            for i in range(L):
                cur = seq[i]
                options = branching_maps[i].get(cur, set())
                bs = len(options)
                if bs == 0:
                    # degenerate: no outgoing alternative observed; set zero probability
                    prob = 0.0
                    break
                prob *= 1.0 / bs
            probs.append(prob)

        total = sum(probs)
        if total <= 0.0:
            # nothing reachable or numerical underflow; return uniform distribution over non-zero paths
            nonzero = [p for p in probs if p > 0]
            if not nonzero:
                # no valid path
                return [0.0 for _ in probs]
            # normalize by count of non-zero
            norm = len(nonzero)
            return [(1.0 / norm) if p > 0 else 0.0 for p in probs]

        # normalize
        probs = [p / total for p in probs]
        # safety: ensure sum to 1.0 within tolerance
        s = sum(probs)
        if abs(s - 1.0) > 1e-9:
            # renormalize
            probs = [p / s for p in probs]
        return probs

    def __repr__(self):
        return (f"SimpleFatTreeModel(k={self.k}, hosts={self.hosts_count}, "
                f"edges={self.edge_count}, aggregates={self.aggregate_count}, "
                f"cores={self.core_count}, pods={self.pods_count}), links={self.links}")


from collections import Counter

links = [
    FTDirectedLink(LinkType.H_E, ((LevelType.HOST, 7), (LevelType.EDGE, 3))),
    FTDirectedLink(LinkType.H_E, ((LevelType.EDGE, 3), (LevelType.HOST, 7))),  # same key
]
c = Counter(links)
print(c[links[0]])  # 2

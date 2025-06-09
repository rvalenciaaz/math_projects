#!/usr/bin/env python3
"""
Align all pairs of GraphML graphs contained in a directory.
For every unique unordered pair (G1, G2) the script attempts to find a
node‑label mapping that maximises the number of common edges using a simple
hill‑climbing heuristic with multiple random restarts (the same procedure from
*align.py*). For each pair it produces

* three PNG visualisations (symmetric difference, original G1 with removals,
  remapped G2 with additions)
* the combined symmetric‑difference GraphML
* relabelled original G1 and remapped G2 GraphML files
* a plain‑text log with the node mapping and statistics

All results for a pair are stored in a dedicated sub‑directory inside the user
specified output directory.

Usage
-----
$ ./align_folder.py /path/to/dir-with-graphml [options]

Options
~~~~~~~
--restarts INT     Number of random restarts for the hill‑climber (default 100)
--seed INT         PRNG seed (default 42)
--suffix EXT       File‑name suffix to consider as graphs (default ".graphml")
--skip-size-mismatch   Skip pairs whose node counts differ (default False)
--outdir DIR       Directory where pair sub‑folders are created (default "./results")
--no-plots         Disable PNG plot generation (still writes GraphML) (default False)

Example
~~~~~~~
$ ./align_folder.py graphs/ --restarts 250 --outdir results/
"""

import os
import itertools
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import networkx as nx

# -----------------------------------------------------------------------------
#  Core heuristic alignment (identical to align.py)
# -----------------------------------------------------------------------------

def evaluate_mapping(G1_edges, G2_edges, mapping):
    """Count common edges between *G1_edges* and *G2_edges* under *mapping*."""
    mapped = {tuple(sorted((mapping[u], mapping[v]))) for u, v in G2_edges}
    return len(G1_edges & mapped), mapped


def local_swap_search(nodes, G1_edges, G2_edges, init_map):
    """Hill‑climbing that swaps node labels to increase common edge count."""
    mapping = init_map.copy()
    best_common, _ = evaluate_mapping(G1_edges, G2_edges, mapping)
    improved = True
    while improved:
        improved = False
        pairs = list(itertools.combinations(nodes, 2))
        random.shuffle(pairs)
        for i, j in pairs:
            new_map = mapping.copy()
            new_map[i], new_map[j] = mapping[j], mapping[i]
            common, _ = evaluate_mapping(G1_edges, G2_edges, new_map)
            if common > best_common:
                best_common, mapping = common, new_map
                improved = True
                break
    return best_common, mapping, evaluate_mapping(G1_edges, G2_edges, mapping)[1]


def find_best_alignment_heuristic(G1, G2, *, restarts=100, seed=42):
    """Run *restarts* hill‑climbing restarts to find a good alignment."""
    random.seed(seed)
    nodes = sorted(G1.nodes())
    G1_edges = {tuple(sorted(e)) for e in G1.edges()}
    G2_edges = list(G2.edges())
    best_common = -1
    best_map = None
    best_mapped = None
    for _ in trange(restarts, desc="Restarts", leave=False):
        perm = nodes.copy()
        random.shuffle(perm)
        init_map = dict(zip(nodes, perm))
        common, mapping, mapped = local_swap_search(nodes, G1_edges, G2_edges, init_map)
        if common > best_common:
            best_common, best_map, best_mapped = common, mapping, mapped
    return best_common, best_map, best_mapped

# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------

def maybe_int(x):
    """Attempt to convert *x* to int, otherwise return original."""
    try:
        return int(x)
    except ValueError:
        return x


def normalise_labels(G):
    """Relabel nodes to ints where possible for consistent layouts."""
    mapping = {n: maybe_int(n) for n in G.nodes()}
    return nx.relabel_nodes(G, mapping)


# -----------------------------------------------------------------------------
#  Alignment + persistence for one pair
# -----------------------------------------------------------------------------

def align_pair(g1_path: Path, g2_path: Path, *, args):
    pair_name = f"{g1_path.stem}_{g2_path.stem}"
    pair_out = args.outdir / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    # Load graphs
    G1 = normalise_labels(nx.read_graphml(g1_path))
    G2 = normalise_labels(nx.read_graphml(g2_path))

    if args.skip_size_mismatch and len(G1) != len(G2):
        print(f"[SKIP] {pair_name}: node counts differ ({len(G1)} vs {len(G2)})")
        return

    # Heuristic alignment
    best_common, mapping, mapped_edges = find_best_alignment_heuristic(
        G1, G2, restarts=args.restarts, seed=args.seed
    )

    # Save mapping log
    mapping_txt = pair_out / "mapping.txt"
    with mapping_txt.open("w") as fh:
        fh.write(f"Best common edges: {best_common}\n")
        fh.write("Node mapping (G2 -> G1):\n")
        for g2, g1 in mapping.items():
            fh.write(f"  {g2} -> {g1}\n")
    print(f"[OK] {pair_name}: common={best_common}  → saved {mapping_txt.name}")

    # Prepare edge sets for visualisation / GraphML output
    E1 = {tuple(sorted(e)) for e in G1.edges()}
    to_remove = list(E1 - mapped_edges)
    to_add = list(mapped_edges - E1)
    common_edges = list(E1 & mapped_edges)

    # Remap G2
    G2m = nx.relabel_nodes(G2, mapping)

    # Combine graph for symmetric‑diff (same nodes as G1)
    H = nx.Graph()
    H.add_nodes_from(G1.nodes())
    H.add_edges_from(to_remove + to_add)

    # Consistent layout once per pair
    pos = nx.spring_layout(H, seed=args.seed)

    # ---------------------------------------------------------------------
    #  Visualisations
    # ---------------------------------------------------------------------
    if not args.no_plots:
        # Symmetric difference overlay
        plt.figure(figsize=(6, 6))
        nx.draw_networkx_nodes(H, pos, node_color="lightblue")
        nx.draw_networkx_labels(H, pos)
        nx.draw_networkx_edges(H, pos, edgelist=to_remove, edge_color="red", width=2)
        nx.draw_networkx_edges(H, pos, edgelist=to_add, edge_color="green", width=2)
        plt.title("Symmetric Difference (Red=Remove, Green=Add)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(pair_out / "symmetric_diff.png", dpi=300)
        plt.close()

        # Original G1 with removals
        plt.figure(figsize=(6, 6))
        nx.draw(G1, pos, labels={n: n for n in G1.nodes()}, node_color="lightblue",
                edgelist=common_edges, width=1)
        nx.draw_networkx_edges(G1, pos, edgelist=to_remove, edge_color="red", width=2)
        plt.title("Original G1 (Red=Edges to Remove)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(pair_out / "G1_original.png", dpi=300)
        plt.close()

        # Remapped G2 with additions
        plt.figure(figsize=(6, 6))
        nx.draw(G2m, pos, labels={n: n for n in G2m.nodes()}, node_color="lightblue",
                edgelist=common_edges, width=1)
        nx.draw_networkx_edges(G2m, pos, edgelist=to_add, edge_color="green", width=2)
        plt.title("Remapped G2 (Green=Edges to Add)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(pair_out / "G2_remapped.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------------------
    #  GraphML persistence
    # ---------------------------------------------------------------------
    nx.write_graphml(H, pair_out / "symmetric_diff.graphml")
    nx.write_graphml(G1, pair_out / "G1_relabelled.graphml")
    nx.write_graphml(G2m, pair_out / "G2_remapped.graphml")

# -----------------------------------------------------------------------------
#  Main CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Align all pairs of GraphML graphs in a folder")
    p.add_argument("directory", type=Path, help="Folder containing GraphML files to align")
    p.add_argument("--restarts", type=int, default=100, help="Random restarts for hill‑climber (default: 100)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--suffix", default=".graphml", help="GraphML file suffix (default: .graphml)")
    p.add_argument("--skip-size-mismatch", action="store_true", help="Skip pairs with differing node counts")
    p.add_argument("--outdir", type=Path, default=Path("./results"), help="Base output directory (default: ./results)")
    p.add_argument("--no-plots", action="store_true", help="Disable PNG plot generation to save time")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.directory.is_dir():
        raise SystemExit(f"Input directory '{args.directory}' does not exist or is not a directory")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Collect GraphML files
    files = sorted([p for p in args.directory.iterdir() if p.suffix == args.suffix])
    if len(files) < 2:
        raise SystemExit("Need at least two GraphML files to perform pairwise alignment")

    print(f"Found {len(files)} graph files – aligning {len(files)*(len(files)-1)//2} pairs...\n")

    for g1_path, g2_path in tqdm(list(itertools.combinations(files, 2)), desc="Pairs"):
        align_pair(g1_path, g2_path, args=args)

    print("\nAll done! Results written to:", args.outdir.resolve())


if __name__ == "__main__":
    main()

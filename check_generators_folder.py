#!/usr/bin/env python3
"""
find_generators.py

Traverse sub-directories, read the two GraphML files that define a pair of
graphs (original + swapped), compute the perturbation matrix M, identify the
zero-row vertices, and print a minimal generator set for the group 𝒵
described in condition C-4.

Requires:  networkx  (pip install networkx)
"""

import os
import argparse
from itertools import islice
import numpy as np
import networkx as nx


# ──────────────────────────────────────────────────────────── helpers ──
def read_graph(path):
    """
    Load a GraphML file and return an undirected simple graph
    (no parallel edges, no self-loops).
    """
    H = nx.read_graphml(path)          # may be DiGraph / MultiGraph
    G = nx.Graph(H)                    # collapses parallel edges, makes undirected
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def adjacency_matrix(G, nodelist):
    """Return the adjacency matrix as a dense NumPy array with 0 / 1 entries."""
    return nx.to_numpy_array(G, nodelist=nodelist, dtype=int)


def generators_from_M(M):
    """
    Return
      * zero_rows – list of vertex indices whose rows of M are identically zero,
      * gens – list of (u, v) transpositions that form a deterministic
               generating set: (0↔1), (2↔3), … in the zero_row ordering.
    """
    zero_rows = np.where(~M.any(axis=1))[0].tolist()
    gens = [
        (zero_rows[i], zero_rows[i + 1])
        for i in range(0, len(zero_rows) // 2 * 2, 2)
    ]
    return zero_rows, gens


def process_folder(folder_path):
    """Compute zero-row vertices and generators for one experiment folder."""
    g1_file = os.path.join(folder_path, "G1_relabelled.graphml")
    g2_file = os.path.join(folder_path, "G2_remapped.graphml")
    if not (os.path.exists(g1_file) and os.path.exists(g2_file)):
        return None

    G1 = read_graph(g1_file)
    G2 = read_graph(g2_file)

    # Consistent node order (string-sorted to avoid “1, 10, 2” pitfalls)
    nodes = sorted(G1.nodes(), key=str)

    A0 = adjacency_matrix(G1, nodes)
    A1 = adjacency_matrix(G2, nodes)
    M = A1 - A0

    zero_rows, generators = generators_from_M(M)
    return {
        "folder": os.path.basename(folder_path),
        "nodes": nodes,
        "zero_rows": zero_rows,
        "generators": generators,
    }


# ───────────────────────────────────────────────────────────── main ──
def main(parent_dir):
    for entry in sorted(os.listdir(parent_dir)):
        sub = os.path.join(parent_dir, entry)
        if not os.path.isdir(sub):
            continue

        result = process_folder(sub)
        if result is None:
            continue

        # Human-friendly printing
        print(f"\n📂  {result['folder']}")
        print("   zero-row vertices:",
              [result["nodes"][i] for i in result["zero_rows"]] or "∅")

        if result["generators"]:
            gens = ", ".join(
                f"({result['nodes'][u]} {result['nodes'][v]})"
                for u, v in result["generators"]
            )
            print("   generators:        ", gens)
        else:
            print("   generators:         <none>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find generators produced by double-edge swaps "
                    "across many experiment folders.")
    parser.add_argument(
        "parent",
        help="Path to the directory whose immediate sub-folders "
             "contain the two GraphML files.")
    args = parser.parse_args()
    main(args.parent)

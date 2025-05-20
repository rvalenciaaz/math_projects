#!/usr/bin/env python3
"""
batch_print_spectra.py

Load all GraphML graphs from a directory, search for k-sized double-edge swap families, and for each one print the
eigenvalues before and after the swap in a nicely formatted way.
Removed outside balance and outside orthogonality conditions.
"""

import argparse
import itertools
from pathlib import Path

import networkx as nx
import numpy as np


def load_graph(path):
    G = nx.read_graphml(path)
    return G if isinstance(G, nx.Graph) else nx.Graph(G)


def candidate_blocks(G):
    """All 4-sets inducing exactly two independent edges."""
    cands = []
    for e, f in itertools.combinations(G.edges(), 2):
        u, v = e; x, y = f
        if {u, v} & {x, y}:
            continue
        M = {u, v, x, y}
        # ensure no other edges among M
        extra = any(
            G.has_edge(a, b)
            for a, b in itertools.combinations(M, 2)
            if {a, b} not in ({u, v}, {x, y})
        )
        if not extra:
            cands.append({'nodes': M, 'e': (u, v), 'f': (x, y)})
    return cands


def block_block_ok(G, family):
    for c1, c2 in itertools.combinations(family, 2):
        u1, v1 = c1['e']; x1, y1 = c1['f']
        u2, v2 = c2['e']; x2, y2 = c2['f']
        P1, N1 = {u1, v1}, {x1, y1}
        P2, N2 = {u2, v2}, {x2, y2}

        def cnt(A, B):
            return sum(1 for a in A for b in B if G.has_edge(a, b))

        if cnt(P1, P2) != cnt(N1, N2):
            return False
        if cnt(P1, N2) != cnt(N1, P2):
            return False

    return True


def swap_graph(G, family):
    Gp = G.copy()
    for c in family:
        u, v = c['e']; x, y = c['f']
        Gp.remove_edge(u, v)
        Gp.remove_edge(x, y)
        Gp.add_edge(u, x)
        Gp.add_edge(v, y)
    return Gp


def sorted_spectrum(G):
    A = nx.to_numpy_array(G)
    vals = np.linalg.eigvals(A).real
    idx = np.lexsort((vals, np.abs(vals)))
    return vals[idx]


def format_spectrum(spec, precision=6, per_line=8):
    """Format an array of eigenvalues into a multi-line string with given precision."""
    formatted = [f"{v:.{precision}f}" for v in spec]
    lines = []
    for i in range(0, len(formatted), per_line):
        lines.append("  " + "  ".join(formatted[i:i+per_line]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Load all GraphML files from a directory and print eigenvalues before/after swaps"
    )
    parser.add_argument(
        '-I', '--input-dir', required=True,
        help="Directory containing .graphml files"
    )
    parser.add_argument(
        '-k', '--k', type=int, required=True,
        help="Size of the swap family"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"Input directory '{input_dir}' does not exist or is not a directory.")

    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() != '.graphml':
            continue

        print(f"\n--- Processing {path.name} ---")
        G = load_graph(path)
        orig_spec = sorted_spectrum(G)
        cands = candidate_blocks(G)
        print(f"Found {len(cands)} candidate blocks.")

        found = 0
        for combo in itertools.combinations(cands, args.k):
            used = set()
            if any((used & c['nodes']) or used.update(c['nodes']) for c in combo):
                continue

            if not block_block_ok(G, combo):
                continue

            Gp = swap_graph(G, combo)
            new_spec = sorted_spectrum(Gp)
            if not np.allclose(np.abs(orig_spec), np.abs(new_spec), atol=1e-8):
                continue

            found += 1
            print(f"\n=== Fully Block-Balanced Family #{found} in {path.name} ===")
            print("Original eigenvalues:")
            print(format_spectrum(orig_spec))
            print("Swapped eigenvalues:")
            print(format_spectrum(new_spec))

        if found == 0:
            print(f"No fully balanced families of size k={args.k} found in {path.name}.")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
find_fully_balanced_folder.py

Batch‐search GraphML files for fully balanced swaps, with tqdm progress bars.
"""

import argparse
import itertools
import math
import os
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

def load_graph(path):
    G = nx.read_graphml(path)
    return G if isinstance(G, nx.Graph) else nx.Graph(G)

def candidate_blocks(G):
    """All 4‐sets inducing exactly two independent edges."""
    cands = []
    for e, f in itertools.combinations(G.edges(), 2):
        u, v = e; x, y = f
        if {u, v} & {x, y}:
            continue
        M = {u, v, x, y}
        extra = any(
            G.has_edge(a, b)
            for a, b in itertools.combinations(M, 2)
            if {a, b} not in ({u, v}, {x, y})
        )
        if not extra:
            cands.append({'nodes': M, 'e': (u, v), 'f': (x, y)})
    return cands

def build_delta(G, family):
    adj = {w: set(G[w]) for w in G}
    used = set().union(*(c['nodes'] for c in family))
    outside = sorted(w for w in G if w not in used)
    Δ = []
    for w in outside:
        row = []
        for c in family:
            u, v = c['e']; x, y = c['f']
            row.append(len(adj[w] & {u, v}) - len(adj[w] & {x, y}))
        Δ.append(row)
    return outside, np.array(Δ, dtype=int)

def is_balanced(Δ):
    return np.all(Δ.sum(axis=1) == 0)

def is_orthogonal(Δ):
    k = Δ.shape[1]
    for i, j in itertools.combinations(range(k), 2):
        if Δ[:, i].dot(Δ[:, j]) != 0:
            return False
    return True

def block_block_ok(G, family):
    for c1, c2 in itertools.combinations(family, 2):
        u1, v1 = c1['e']; x1, y1 = c1['f']
        u2, v2 = c2['e']; x2, y2 = c2['f']
        P1, N1 = {u1, v1}, {x1, y1}
        P2, N2 = {u2, v2}, {x2, y2}

        def cnt(A, B):
            return sum(1 for a in A for b in B if G.has_edge(a, b))

        if cnt(P1, P2) != cnt(N1, N2): return False
        if cnt(P1, N2) != cnt(N1, P2): return False

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

def process_graph(path, k):
    G = load_graph(path)
    orig_spec = sorted_spectrum(G)
    cands = candidate_blocks(G)

    # prepare tqdm over combinations
    total = math.comb(len(cands), k) if len(cands) >= k else 0
    found = 0

    for combo in tqdm(
        itertools.combinations(cands, k),
        desc=f"{path.name} combos",
        total=total,
        leave=False
    ):
        # disjointness
        used = set()
        if any((used & c['nodes']) or used.update(c['nodes']) for c in combo):
            continue

        _, Δ = build_delta(G, combo)
        if not is_balanced(Δ) or not is_orthogonal(Δ):
            continue
        if not block_block_ok(G, combo):
            continue

        Gp = swap_graph(G, combo)
        new_spec = sorted_spectrum(Gp)
        if not np.allclose(np.abs(orig_spec), np.abs(new_spec), atol=1e-8):
            continue

        found += 1

    return found

def main():
    parser = argparse.ArgumentParser(
        description="Batch‐search GraphML files for fully balanced swaps"
    )
    parser.add_argument(
        '-I', '--input-dir', required=True,
        help="Directory containing .graphml files"
    )
    parser.add_argument(
        '-O', '--output-dir', required=True,
        help="Directory to copy ‘good’ graphs into"
    )
    parser.add_argument(
        '-k', '--k', type=int, required=True,
        help="Size of the swap family"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # wrap the file loop in tqdm
    for path in tqdm(list(input_dir.iterdir()), desc="Graph files"):
        if path.suffix.lower() != '.graphml':
            continue

        print(f"\nProcessing {path.name}...", end=' ')
        try:
            count = process_graph(path, args.k)
        except Exception as e:
            print(f"error: {e}")
            continue

        if count > 0:
            shutil.copy2(path, output_dir / path.name)
            print(f"FOUND ({count} families) → copied")
        else:
            print("none found")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""cospectral.py

Find cospectral pairs of vertices and the cospectral classes they induce in two input
GraphML graphs.

Usage
-----
$ python cospectral.py graph1.graphml graph2.graphml [--tol 1e-6]

The script prints, for each graph, every pair of vertices whose vertex-deleted
spectra are identical (within the supplied numeric tolerance) and the maximal
cospectral classes (equivalence classes of mutually‑cospectral vertices).
It then lists all cross‑graph cospectral pairs – vertices from different graphs
that share the same vertex‑deleted spectrum.

Dependencies
------------
Python ≥3.8, NetworkX ≥3.0, NumPy ≥1.20.
Install them with:
    pip install networkx numpy
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np


###############################################################################
# Helper functions
###############################################################################

def _canonical_spectrum(evals: np.ndarray, *, decimals: int = 8) -> tuple[float, ...]:
    """Return a rounded, sorted tuple that can serve as a stable key."""
    real_evals = np.real_if_close(evals, tol=1e-6)  # strip negligible imaginary part
    rounded = np.round(np.sort(real_evals), decimals=decimals)
    return tuple(float(x) for x in rounded)


def _vertex_signature(G: nx.Graph, v, *, tol: float) -> tuple[float, ...]:
    """Spectrum signature of G with vertex *v* deleted."""
    G_sub = G.copy()
    G_sub.remove_node(v)
    A = nx.to_numpy_array(G_sub)
    evals = np.linalg.eigvals(A)
    return _canonical_spectrum(evals)


def _signatures_by_vertex(G: nx.Graph, *, tol: float) -> dict:
    """Map vertex -> spectrum signature (tuple)."""
    return {v: _vertex_signature(G, v, tol=tol) for v in G.nodes()}


def _invert_signatures(sig_map: dict) -> dict:
    """Invert vertex→signature to signature→[vertices]."""
    inv: dict[tuple[float, ...], list] = defaultdict(list)
    for v, sig in sig_map.items():
        inv[sig].append(v)
    return inv


def _cospectral_pairs(sig_inv: dict) -> list[tuple]:
    """Return all unordered vertex pairs that are cospectral."""
    pairs: list[tuple] = []
    for verts in sig_inv.values():
        if len(verts) > 1:
            verts_sorted = sorted(verts)
            pairs.extend(itertools.combinations(verts_sorted, 2))
    return pairs


def _cospectral_classes(sig_inv: dict) -> list[list]:
    """Return a list of cospectral classes (size ≥ 2)."""
    return [sorted(verts) for verts in sig_inv.values() if len(verts) > 1]


###############################################################################
# Main logic
###############################################################################

def analyse_graph(G: nx.Graph, *, tol: float):
    sig_map = _signatures_by_vertex(G, tol=tol)
    sig_inv = _invert_signatures(sig_map)
    pairs = _cospectral_pairs(sig_inv)
    classes = _cospectral_classes(sig_inv)
    return sig_map, pairs, classes


def print_report(name: str, pairs: list[tuple], classes: list[list]):
    print("=" * 80)
    print(f"{name} – Cospectral Analysis")
    print("=" * 80)

    if pairs:
        print("Cospectral pairs (|P| = {}):".format(len(pairs)))
        for u, v in pairs:
            print(f"  ({u}, {v})")
    else:
        print("No cospectral pairs found.")

    print()

    if classes:
        print("Cospectral classes (|C| = {}):".format(len(classes)))
        for i, cls in enumerate(classes, start=1):
            members = ", ".join(map(str, cls))
            print(f"  Class {i} (size {len(cls)}): {members}")
    else:
        print("No non‑trivial cospectral classes found.")

    print()


def print_cross_pairs(name1: str, name2: str, sig_inv1: dict, sig_inv2: dict):
    common = set(sig_inv1) & set(sig_inv2)
    cross_pairs: list[tuple[str, str]] = []
    for sig in common:
        for u in sig_inv1[sig]:
            for v in sig_inv2[sig]:
                cross_pairs.append((u, v))

    print("=" * 80)
    print(f"Cross‑graph cospectral pairs between {name1} and {name2}")
    print("=" * 80)
    if cross_pairs:
        print(f"Total cross pairs: {len(cross_pairs)}")
        for u, v in cross_pairs:
            print(f"  ({name1}:{u}, {name2}:{v})")
    else:
        print("No cross‑graph cospectral pairs found.")
    print()


###############################################################################
# Entry point
###############################################################################

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Find cospectral pairs and classes in two GraphML graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("graph1", type=Path, help="Path to first .graphml file")
    parser.add_argument("graph2", type=Path, help="Path to second .graphml file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Eigenvalue equality tolerance (absolute)",
    )
    args = parser.parse_args(argv)

    # Read graphs and convert to simple undirected graphs.
    G1 = nx.Graph(nx.read_graphml(args.graph1))
    G2 = nx.Graph(nx.read_graphml(args.graph2))

    sig_map1, pairs1, classes1 = analyse_graph(G1, tol=args.tol)
    sig_map2, pairs2, classes2 = analyse_graph(G2, tol=args.tol)

    # Print per‑graph reports.
    print_report(args.graph1.name, pairs1, classes1)
    print_report(args.graph2.name, pairs2, classes2)

    # Print cross‑graph cospectral vertices.
    sig_inv1 = _invert_signatures(sig_map1)
    sig_inv2 = _invert_signatures(sig_map2)
    print_cross_pairs(args.graph1.name, args.graph2.name, sig_inv1, sig_inv2)


if __name__ == "__main__":
    main()

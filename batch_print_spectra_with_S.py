#!/usr/bin/env python3
"""
batch_print_spectra_with_S_check.py

Load all GraphML graphs from a directory, search for k-sized fully block-balanced
double-edge-swap families, print the eigenvalues before/after the swaps, and for
each valid family construct the involution S, verify S·A·S = A', and report success.
"""

import argparse
import itertools
from pathlib import Path

import networkx as nx
import numpy as np


def load_graph(path):
    """Load a GraphML file into a NetworkX Graph."""
    G = nx.read_graphml(path)
    return G if isinstance(G, nx.Graph) else nx.Graph(G)


def candidate_blocks(G):
    """Return all 4‐tuples inducing exactly two independent edges."""
    cands = []
    for e, f in itertools.combinations(G.edges(), 2):
        u, v = e
        x, y = f
        # disjoint edges
        if {u, v} & {x, y}:
            continue
        M = {u, v, x, y}
        # no extra edges among the four
        extra = any(
            G.has_edge(a, b)
            for a, b in itertools.combinations(M, 2)
            if {a, b} not in ({u, v}, {x, y})
        )
        if not extra:
            cands.append({'nodes': M, 'e': (u, v), 'f': (x, y)})
    return cands


def block_block_ok(G, family):
    """Check the fully block‐balanced condition for a family of blocks."""
    for c1, c2 in itertools.combinations(family, 2):
        u1, v1 = c1['e']
        x1, y1 = c1['f']
        u2, v2 = c2['e']
        x2, y2 = c2['f']
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
    """Perform the k‐double‐edge swap on G, returning a new graph Gp."""
    Gp = G.copy()
    for c in family:
        u, v = c['e']
        x, y = c['f']
        Gp.remove_edge(u, v)
        Gp.remove_edge(x, y)
        Gp.add_edge(u, x)
        Gp.add_edge(v, y)
    return Gp


def compute_involution_S(ordering, family):
    """
    Given a consistent node ordering and a family of k blocks,
    return the diagonal involution S (as an (n,n) numpy array)
    with -1 on each block‐node and +1 elsewhere.
    """
    n = len(ordering)
    idx = {v: i for i, v in enumerate(ordering)}
    S = np.eye(n, dtype=int)
    # Mark exactly the 2k block‐vertices with -1
    for block in family:
        u, v = block['e']
        x, y = block['f']
        for w in (u, v, x, y):
            S[idx[w], idx[w]] = -1
    return S


def sorted_spectrum(G):
    """Return the real eigenvalues of A(G), sorted by (value, absolute value)."""
    A = nx.to_numpy_array(G)
    vals = np.linalg.eigvals(A).real
    idx = np.lexsort((vals, np.abs(vals)))
    return vals[idx]


def format_spectrum(spec, precision=6, per_line=8):
    """Format a list of floats into aligned lines of fixed‐precision numbers."""
    formatted = [f"{v:.{precision}f}" for v in spec]
    lines = []
    for i in range(0, len(formatted), per_line):
        lines.append("  " + "  ".join(formatted[i:i+per_line]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Load GraphMLs, find fully block‐balanced k‐swap families, "
                    "print spectra, and verify S·A·S = A'"
    )
    parser.add_argument('-I', '--input-dir', required=True,
                        help="Directory containing .graphml files")
    parser.add_argument('-k', '--k', type=int, required=True,
                        help="Size of the swap family (number of blocks)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"Input directory '{input_dir}' does not exist or is not a directory.")

    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() != '.graphml':
            continue

        print(f"\n--- Processing {path.name} ---")
        G = load_graph(path)
        ordering = list(G.nodes())
        A = nx.to_numpy_array(G, nodelist=ordering, dtype=int)
        orig_spec = sorted_spectrum(G)

        cands = candidate_blocks(G)
        print(f"Found {len(cands)} candidate blocks.")

        found = 0
        for combo in itertools.combinations(cands, args.k):
            # ensure disjointness
            used = set()
            if any((used & c['nodes']) or used.update(c['nodes']) for c in combo):
                continue
            # check full block balance
            if not block_block_ok(G, combo):
                continue

            # perform the swap
            Gp = swap_graph(G, combo)
            new_spec = sorted_spectrum(Gp)
            # require singularly cospectral
            if not np.allclose(np.abs(orig_spec), np.abs(new_spec), atol=1e-8):
                continue

            found += 1
            print(f"\n=== Fully Block‐Balanced Family #{found} ===")

            # 1) Build the involution S
            S = compute_involution_S(ordering, combo)

            # 2) Compute S·A·S
            SAS = S.dot(A).dot(S)

            # 3) Extract A' from the swapped graph
            Aprime = nx.to_numpy_array(Gp, nodelist=ordering, dtype=int)

            # 4) Verify S·A·S == A'
            if np.array_equal(SAS, Aprime):
                print("✓ Verified S·A·S = A' for this family.")
            else:
                print("✗ Verification failed: S·A·S != A'!")
                # Report mismatch
                diff = SAS - Aprime
                nz = np.argwhere(diff != 0)
                print("First mismatches (up to 10):")
                for i, j in nz[:10]:
                    print(f"  ({ordering[i]}, {ordering[j]}): SAS={SAS[i,j]}, A'={Aprime[i,j]}")

            # Print spectra
            print("Original eigenvalues:")
            print(format_spectrum(orig_spec))
            print("Swapped eigenvalues:")
            print(format_spectrum(new_spec))

        if found == 0:
            print(f"No fully block‐balanced families of size k={args.k} found in {path.name}.")


if __name__ == '__main__':
    main()

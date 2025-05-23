#!/usr/bin/env python3
"""
Standalone script to compare an original graph and its “swapped” variant:
  - Loads two GraphML files specified on the command line.
  - Sorts nodes numerically (if possible) for consistent ordering.
  - Builds adjacency matrices A_orig, A_swap.
  - Prints the difference D = A_swap - A_orig and its element‐wise square D².
  - Computes eigenvalues and absolute‐value spectra.
  - Performs SVD to get U, Σ, V^T for each.
  - Constructs bi‐orthogonal maps O_L = U_swap @ U_orig^T and
    O_R = V_swap @ V_orig^T.
  - Computes reconstruction and orthonormality errors.
  - Finds a permutation matrix P so that O_R ≈ O_L @ P.
  - Prints all results to stdout, with nicely formatted matrices.
"""

import argparse
import sys

import numpy as np
import networkx as nx
import scipy.linalg as la
import numpy.linalg as nal

def load_graph_adj(path):
    G = nx.read_graphml(path)
    # Attempt numeric sort of node labels, else fallback to lexicographic
    try:
        nodes = sorted(G.nodes(), key=lambda x: int(x))
    except ValueError:
        nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    return A, nodes

def compute_spectra(A):
    eigs = np.linalg.eigvals(A)
    return np.sort(eigs.real), np.sort(np.abs(eigs.real))

def compute_svd_maps(A_orig, A_swap):
    # full SVD
    U1, S1, V1t = la.svd(A_orig)
    U2, S2, V2t = la.svd(A_swap)
    # bi-orthogonal maps
    O_L = U2 @ U1.T
    O_R = V2t.T @ V1t
    return O_L, O_R, S1, S2

def frobenius_error(X):
    return np.linalg.norm(X, ord='fro')

def find_permutation(O_L, O_R, tol=1e-6):
    """
    Find P (permutation matrix) such that O_R ≈ O_L @ P.
    Match each O_R column to the O_L column with highest absolute inner product.
    """
    n = O_L.shape[1]
    P = np.zeros((n, n), dtype=int)
    used = set()
    for j in range(n):
        prods = np.abs(O_L.T @ O_R[:, j])
        for i in used:
            prods[i] = -1
        i_max = int(np.argmax(prods))
        if prods[i_max] < 1 - tol:
            print(f"Warning: best match for col {j} has inner prod {prods[i_max]:.3f}", file=sys.stderr)
        P[i_max, j] = 1
        used.add(i_max)
    return P

def pretty_print_matrix(name, M, fmt="{:8.4f}"):
    """
    Print matrix M with aligned columns in the terminal.
    fmt specifies width and precision, e.g. "{:8.4f}".
    """
    rows, cols = M.shape
    print(f"{name} ({rows}×{cols}):")
    for r in range(rows):
        row_str = "  " + " ".join(fmt.format(M[r, c]) for c in range(cols))
        print(row_str)
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Compare an original and swapped graph by eigen- and singular-spectra.")
    parser.add_argument("orig_graph", help="Path to original .graphml")
    parser.add_argument("swap_graph", help="Path to swapped .graphml")
    args = parser.parse_args()

    # Load adjacency matrices (ordered by sorted node IDs)
    A_orig, nodes = load_graph_adj(args.orig_graph)
    A_swap, _     = load_graph_adj(args.swap_graph)

    # Print node order if you want to verify:
    # print("Node order:", nodes)

    # Difference and its element-wise square
    D  = (A_swap - A_orig).astype(int)
    D2 = (-np.matmul(D.T,D)).astype(int)
    D3 = (-np.matmul(D,D.T)).astype(int)

    # Spectra
    eig_orig, abs_orig = compute_spectra(A_orig)
    eig_swap, abs_swap = compute_spectra(A_swap)

    # Print eigenvalue spectra
    print("Eigenvalues (sorted):")
    print("  original:", eig_orig)
    print("  swapped :", eig_swap)
    print()
    print("Absolute eigenvalues (sorted):")
    print("  original:", abs_orig)
    print("  swapped :", abs_swap)
    print()

    # Build SVD-based maps
    O_L, O_R, S1, S2 = compute_svd_maps(A_orig, A_swap)

    # Pretty-print the bi-orthogonal maps
    pretty_print_matrix("O_L (U_swap @ U_orig^T)", O_L)
    pretty_print_matrix("O_R (V_swap @ V_orig^T)", O_R)

    # Reconstruction errors
    rec_err    = frobenius_error(O_L @ A_orig @ O_R.T - A_swap)
    ortho_err_L = frobenius_error(O_L.T @ O_L - np.eye(O_L.shape[0]))
    ortho_err_R = frobenius_error(O_R.T @ O_R - np.eye(O_R.shape[0]))

    print(f"Reconstruction error ||O_L A_orig O_R^T - A_swap||_F: {rec_err:.6e}")
    print(f"O_L orthonormality error ||O_L^T O_L - I||_F        : {ortho_err_L:.6e}")
    print(f"O_R orthonormality error ||O_R^T O_R - I||_F        : {ortho_err_R:.6e}")
    print()

    # Permutation matrix linking O_L to O_R
    P = find_permutation(O_L, O_R)
    pretty_print_matrix("D = A_swap - A_orig", D, fmt="{:2d}")
    pretty_print_matrix("(A_swap - A_orig)^T(P)", D2, fmt="{:2d}")
    pretty_print_matrix("(A_swap - A_orig)(P^T)", D3, fmt="{:2d}")
    pretty_print_matrix("Permutation matrix P", P, fmt="{:2d}")

if __name__ == "__main__":
    main()

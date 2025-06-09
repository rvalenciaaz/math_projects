#!/usr/bin/env python3
"""
plot_qwalk_returns.py  –  Python 3.12-ready
Compute CTQW return probabilities for every node in each GraphML graph,
plot & save

• per-node mean-return dashed lines and labels (with node IDs)
• a thin dashed black baseline at 1 / (n − 1)
• a thin dashed black line at the **global minimum return** for the graph

and write **graph_stats.csv** in the output folder with:

    graph, vertices, edges, min_return

(No legend included.)
"""

from __future__ import annotations
import argparse
import pathlib
import multiprocessing as mp
import csv

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")                       # headless
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# ─────────── CTQW helpers ───────────
def all_return_probs(A: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return an (n, T) array of CTQW return probabilities."""
    w, V = eigh(A)
    W = np.abs(V) ** 2                      # |V_{si}|²
    phases = np.exp(-1j * w[:, None] * times[None, :])
    amps = W @ phases                       # (n, T)
    return np.abs(amps) ** 2                # (n, T)


# ─────────── worker ───────────
def plot_graph(args):
    """
    Load one GraphML file, compute return probabilities, make the plot,
    and return a stats row for the CSV.
    """
    path, times, outdir, fmt = args
    G = nx.read_graphml(path)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    A = nx.to_numpy_array(G, dtype=float)

    probs = all_return_probs(A, times)      # (n, T)
    mean_probs = probs.mean(axis=1)         # mean per node
    min_return = probs.min()                # global minimum

    fig, ax = plt.subplots(figsize=(6, 4))

    # thin frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)
    ax.tick_params(width=0.2)

    # plot each node's time-series and its dashed mean line
    for s in range(n):
        line, = ax.plot(times, probs[s], lw=0.2, alpha=0.8)
        ax.axhline(mean_probs[s],
                   color=line.get_color(), linestyle="--", lw=0.8, alpha=0.8)

    # group means within tolerance and label once with node IDs
    tol = 1e-3
    unique_means: list[float] = []
    groups: dict[float, list[int]] = {}
    for idx, mpv in enumerate(mean_probs):
        # check for existing mean within tolerance
        for um in unique_means:
            if abs(mpv - um) < tol:
                groups[um].append(idx)
                break
        else:
            unique_means.append(mpv)
            groups[mpv] = [idx]

    # annotate each unique mean diagonally to avoid clashing
    x_offset = (times[-1] - times[0]) * 0.01
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_offset = y_range * 0.005
    for mpv in unique_means:
        ids = ','.join(str(i) for i in groups[mpv])
        ax.text(times[-1] + x_offset,
                mpv + y_offset,
                f"{mpv:.3f} [{ids}]",
                va="bottom", ha="left",
                fontsize="xx-small", alpha=0.8,
                rotation=20)

    # baseline 1 / (n-1)
    if n > 1:
        baseline = 1.0 / (n - 1)
        ax.axhline(baseline, color="k", linestyle="--", lw=0.6, alpha=0.6)
        ax.text(times[-1], baseline,
                f"{baseline:.3f}",
                va="center", ha="left",
                fontsize="xx-small", alpha=0.8)

    # global minimum line
    ax.axhline(min_return, color="k", linestyle="--", lw=0.6, alpha=0.6)
    ax.text(times[-1], min_return,
            f"{min_return:.3f}",
            va="center", ha="left",
            fontsize="xx-small", alpha=0.8)

    # labels & formatting
    ax.set_xlabel("Time t")
    ax.set_ylabel(r"$P_{s \to s}(t)$")
    ax.set_title(f"{path.name}  (n={n})")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # save figure
    fname = outdir / f"{path.stem}_returns.{fmt}"
    fig.tight_layout()
    fig.savefig(fname, dpi=600)
    plt.close(fig)

    # row for CSV
    return (path.stem, n, m, min_return)


# ─────────── CLI & dispatch ───────────
def main() -> None:
    pa = argparse.ArgumentParser(
        description=("Plot CTQW return probabilities with per-node means, "
                     "baselines at 1/(n-1) and global minimum, "
                     "and save a per-graph stats CSV (no legend)."))
    pa.add_argument("folder",  type=pathlib.Path,
                    help="Folder containing *.graphml files")
    pa.add_argument("--outdir", default="return_plots", type=pathlib.Path,
                    help="Directory for plots and CSV")
    pa.add_argument("--tmax",   type=float, default=20.0,
                    help="Maximum time")
    pa.add_argument("--steps",  type=int,   default=600,
                    help="Number of time steps")
    pa.add_argument("--cores",  type=int,   default=1,
                    help="Workers: 1 = serial, 0 = all CPU cores, "
                         "negative -> (cpu_count + cores)")
    pa.add_argument("--fmt",    default="png", choices=["png", "pdf", "svg"],
                    help="Figure format")
    args = pa.parse_args()

    gml_files = sorted(args.folder.glob("*.graphml"))
    if not gml_files:
        raise SystemExit(f"[ERROR] no *.graphml files in {args.folder}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, args.tmax, args.steps)
    tasks = [(p, times, args.outdir, args.fmt) for p in gml_files]

    if args.cores == 0:
        workers = mp.cpu_count()
    elif args.cores < 0:
        workers = max(1, mp.cpu_count() + args.cores)
    else:
        workers = max(1, args.cores)

    print(f"[INFO] plotting {len(gml_files)} graphs using {workers} worker(s)…")

    # process graphs
    if workers == 1:
        rows = list(map(plot_graph, tasks))
    else:
        with mp.Pool(workers) as pool:
            rows = pool.map(plot_graph, tasks)

    # write CSV
    csv_path = args.outdir / "graph_stats.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph", "vertices", "edges", "min_return"])
        writer.writerows(rows)

    print(f"[✓] plots saved to {args.outdir.resolve()}")
    print(f"[✓] stats CSV saved to {csv_path.resolve()}")


if __name__ == "__main__":
    main()

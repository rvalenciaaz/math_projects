#!/usr/bin/env python3
"""
plot_qwalk_returns_recursive.py  –  Python 3.12-ready
Compute CTQW return probabilities for every node in each GraphML graph
found in every **sub-directory** of a parent folder, plot & save:

• per-node mean-return dashed lines (labelled with node IDs)
• thin dashed baselines at 1 / (n − 1) and at the global minimum return
• one CSV per sub-directory:  graph_stats.csv   (graph, vertices, edges, min_return)

No legend is included.

Usage
-----
    python plot_qwalk_returns_recursive.py PARENT_DIR [options]

All output goes to the same sub-directory that contained each input file.
"""

from __future__ import annotations
import argparse
import pathlib
import multiprocessing as mp
import csv

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")                               # headless backend
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# ─────────────────────────────── CTQW helpers ──
def all_return_probs(A: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return an (n, T) array of CTQW return probabilities."""
    w, V = eigh(A)
    W = np.abs(V)**2                                # |V_{si}|²
    phases = np.exp(-1j * w[:, None] * times[None])
    amps = W @ phases                               # (n, T)
    return np.abs(amps)**2                          # (n, T)


# ─────────────────────────────── worker ──
def plot_graph(args):
    """Process one GraphML file and return a row for the stats CSV."""
    path, times, outdir, fmt = args

    G = nx.read_graphml(path)
    n, m = G.number_of_nodes(), G.number_of_edges()
    A = nx.to_numpy_array(G, dtype=float)

    probs = all_return_probs(A, times)              # (n, T)
    mean_probs = probs.mean(axis=1)                 # mean per node
    min_return = probs.min()

    fig, ax = plt.subplots(figsize=(6, 4))

    # thin frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)
    ax.tick_params(width=0.2)

    # node trajectories + mean lines
    for s in range(n):
        line, = ax.plot(times, probs[s], lw=0.2, alpha=0.8)
        ax.axhline(mean_probs[s],
                   color=line.get_color(), ls="--", lw=0.8, alpha=0.8)

    # collect equal means (within tol) and label them once
    tol = 1e-3
    unique_means: list[float] = []
    groups: dict[float, list[int]] = {}
    for idx, mpv in enumerate(mean_probs):
        for um in unique_means:
            if abs(mpv - um) < tol:
                groups[um].append(idx)
                break
        else:
            unique_means.append(mpv)
            groups[mpv] = [idx]

    x_off = (times[-1] - times[0]) * 0.01
    y_off = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005
    for mpv in unique_means:
        ids = ",".join(str(i) for i in groups[mpv])
        ax.text(times[-1] + x_off, mpv + y_off,
                f"{mpv:.3f} [{ids}]",
                va="bottom", ha="left", fontsize="xx-small", alpha=0.8, rotation=20)

    # baselines
    if n > 1:
        base = 1.0 / (n - 1)
        ax.axhline(base, color="k", ls="--", lw=0.6, alpha=0.6)
        ax.text(times[-1], base, f"{base:.3f}",
                va="center", ha="left", fontsize="xx-small", alpha=0.8)

    ax.axhline(min_return, color="k", ls="--", lw=0.6, alpha=0.6)
    ax.text(times[-1], min_return, f"{min_return:.3f}",
            va="center", ha="left", fontsize="xx-small", alpha=0.8)

    # cosmetics
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

    return (path.stem, n, m, min_return)


# ─────────────────────────────── per-folder helper ──
def process_folder(folder: pathlib.Path,
                   times: np.ndarray,
                   fmt: str,
                   cores: int) -> None:
    """Analyse all *.graphml files in *folder* and write plots + CSV there."""
    gml_files = sorted(folder.glob("*.graphml"))
    if not gml_files:
        return

    tasks = [(p, times, folder, fmt) for p in gml_files]

    if cores == 0:
        workers = mp.cpu_count()
    elif cores < 0:
        workers = max(1, mp.cpu_count() + cores)
    else:
        workers = max(1, cores)

    print(f"   → {len(gml_files)} graph(s), {workers} worker(s)")

    if workers == 1:
        rows = list(map(plot_graph, tasks))
    else:
        with mp.Pool(workers) as pool:
            rows = pool.map(plot_graph, tasks)

    # CSV
    csv_path = folder / "graph_stats.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph", "vertices", "edges", "min_return"])
        writer.writerows(rows)


# ─────────────────────────────── CLI & dispatch ──
def main() -> None:
    pa = argparse.ArgumentParser(
        description=("For every immediate sub-directory of a parent folder, "
                     "plot CTQW return probabilities for each GraphML graph "
                     "and save figures plus a per-graph stats CSV "
                     "into that same sub-directory (no legend)."))
    pa.add_argument("parent", type=pathlib.Path,
                    help="Parent directory; each sub-folder must contain *.graphml files")
    pa.add_argument("--tmax",  type=float, default=20.0,
                    help="Maximum time (default 20)")
    pa.add_argument("--steps", type=int,   default=600,
                    help="Number of time steps (default 600)")
    pa.add_argument("--cores", type=int,   default=1,
                    help="Workers per sub-folder "
                         "(1 = serial, 0 = all cores, "
                         "negative → (cpu_count + cores))")
    pa.add_argument("--fmt",   default="png", choices=["png", "pdf", "svg"],
                    help="Figure format (default png)")
    args = pa.parse_args()

    if not args.parent.is_dir():
        raise SystemExit(f"[ERROR] {args.parent} is not a directory")

    times = np.linspace(0.0, args.tmax, args.steps)

    # — iterate over experiment sub-folders —
    for sub in sorted(args.parent.iterdir()):
        if not sub.is_dir():
            continue
        print(f"\n📂  {sub.name}")
        process_folder(sub, times, args.fmt, args.cores)

    print("\n[✓] Done.")


if __name__ == "__main__":
    main()

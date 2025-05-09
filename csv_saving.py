#!/usr/bin/env python
"""
NSGA-II over 3‑regular graphs, maximising
  (1) average return probability across all nodes (CTQW),
  (2) algebraic connectivity (λ₂ / n).

This *multirun* variant:
  • runs the algorithm several times with different seeds,
  • saves each run's Pareto‑optimal graphs **in its own sub‑folder**, and
  • produces one global scatter plot that *joins* all points (without
    extra non‑domination filtering).  Each point is labelled
    "<graph‑id>_<run>" exactly as requested.
  • **NEW:** also writes a CSV file (``combined_points.csv``) with the data
    used for the final scatter so the experiment can be replicated later.
"""
from __future__ import annotations
import argparse, random, pathlib, multiprocessing as mp, csv
from itertools import combinations

import numpy as np
import networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from deap import base, creator, tools, algorithms

# ───────── Register DEAP classes ─────────
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

# ────────────────── Quantum‑walk helpers ──────────────────

def ctqw_amplitudes(A: np.ndarray, s: int, times: np.ndarray) -> np.ndarray:
    w, V = eigh(A)
    return V @ (V[s, :, None] * np.exp(-1j * np.outer(w, times)))

def mean_return_prob(A: np.ndarray, times: np.ndarray) -> float:
    n = A.shape[0]
    returns = []
    for s in range(n):
        amps   = ctqw_amplitudes(A, s, times)
        p_ret  = np.abs(amps[s, :]) ** 2
        returns.append(p_ret.mean())
    return float(np.mean(returns))

# ───────────── Secondary objective: algebraic connectivity ─────────────

def algebraic_conn_norm(G: nx.Graph, n: int) -> float:
    if not nx.is_connected(G):
        return 0.0
    L   = nx.laplacian_matrix(G).astype(float).todense()
    λ2  = np.linalg.eigvalsh(L)[1]
    return float(λ2) / n

# ───────────── Genome ↔ Graph helpers ─────────────

def bits_to_graph(bits, edges, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for b, (u, v) in zip(bits, edges):
        if b:
            G.add_edge(u, v)
    return G

def graph_to_bits(G, edges):
    es = set(G.edges())
    return [1 if (u, v) in es or (v, u) in es else 0 for (u, v) in edges]

# ───────────── Bi‑objective evaluation ─────────────

def evaluate(ind, edges, n, times):
    G = bits_to_graph(ind, edges, n)
    A = nx.to_numpy_array(G, dtype=float)

    mr   = mean_return_prob(A, times)
    conn = algebraic_conn_norm(G, n)
    return mr, conn

# ───────────── Build toolbox ─────────────

def make_toolbox(n, seed=None):
    rng   = random.Random(seed)
    edges = list(combinations(range(n), 2))
    times = np.linspace(0, 20, 60)

    tb = base.Toolbox()

    # init: random 3‑regular → bit‑vector
    def init_regular():
        G = nx.random_regular_graph(d=3, n=n, seed=rng)
        return creator.Individual(graph_to_bits(G, edges))

    tb.register("individual", init_regular)
    tb.register("population", tools.initRepeat, list, tb.individual)

    tb.register("evaluate", evaluate, edges=edges, n=n, times=times)

    # mutation: single double‑edge swap (preserves regularity)
    def mutate_regular(ind):
        G = bits_to_graph(ind, edges, n)
        nx.double_edge_swap(G, nswap=1, max_tries=100)
        ind[:] = graph_to_bits(G, edges)
        return (ind,)

    tb.register("mutate", mutate_regular)
    tb.register("select", tools.selNSGA2)
    return tb, edges

# ───────────── Per‑run save helper ─────────────

def save_pareto_run(run_idx: int, hof, edges, n, outdir: pathlib.Path):
    """Save graphs & scatter for a *single* run."""
    run_dir = outdir / f"run_{run_idx+1}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mrs, conns = [], []

    for graph_id, ind in enumerate(hof):
        G        = bits_to_graph(ind, edges, n)
        mr, conn = ind.fitness.values
        mrs.append(mr); conns.append(conn)

        base_name = f"{graph_id}_{run_idx+1}_ret{mr:.3f}_conn{conn:.3f}"
        nx.write_graphml(G, run_dir / f"{base_name}.graphml")

        plt.figure(figsize=(4, 3))
        nx.draw_spring(G, node_size=80, with_labels=False)
        plt.title(f"⟨P_ret⟩={mr:.3f}, λ₂/n={conn:.3f}")
        plt.tight_layout()
        plt.savefig(run_dir / f"{base_name}.png", dpi=150)
        plt.close()

    # Per‑run scatter (optional but handy)
    plt.figure(figsize=(5, 4))
    plt.scatter(conns, mrs)
    for g, (c, m) in enumerate(zip(conns, mrs)):
        plt.annotate(f"{g}_{run_idx+1}", xy=(c, m), xytext=(2, 2),
                     textcoords="offset points", fontsize=7)
    plt.xlabel("Algebraic connectivity (λ₂/n)")
    plt.ylabel("Mean return probability")
    plt.title(f"Run {run_idx+1}: Pareto front")
    plt.tight_layout()
    plt.savefig(run_dir / "pareto_scatter_labeled.png", dpi=150)
    plt.close()

    # Return the data for the global scatter
    return [(mr, conn, graph_id, run_idx+1) for graph_id, mr, conn in zip(range(len(hof)), mrs, conns)]

# ───────────── Global scatter & CSV helpers ─────────────

def save_combined_scatter(points, outdir: pathlib.Path):
    """Plot every Pareto point from all runs on one scatter."""
    if not points:
        return
    mrs   = [p[0] for p in points]
    conns = [p[1] for p in points]
    labels = [f"{p[2]}_{p[3]}" for p in points]  # "graphid_run"

    plt.figure(figsize=(6, 5))
    plt.scatter(conns, mrs)
    for (c, m, lab) in zip(conns, mrs, labels):
        plt.annotate(lab, xy=(c, m), xytext=(3, 3),
                     textcoords="offset points", fontsize=7)
    plt.xlabel("Algebraic connectivity (λ₂/n)")
    plt.ylabel("Mean return probability")
    plt.title("All runs: Connectivity vs Mean Return")
    plt.tight_layout()
    plt.savefig(outdir / "combined_scatter_labeled.png", dpi=150)
    plt.close()


def save_combined_csv(points, outdir: pathlib.Path):
    """Write a CSV (mean_return, conn_norm, graph_id, run) for all points."""
    if not points:
        return
    csv_path = outdir / "combined_points.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_return", "conn_norm", "graph_id", "run"])
        writer.writerows(points)

# ───────────── Main ─────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--nodes",  type=int,   default=48)
    pa.add_argument("--gens",   type=int,   default=800)
    pa.add_argument("--mu",     type=int,   default=120)
    pa.add_argument("--lam",    type=int,   default=360)
    pa.add_argument("--mutpb",  type=float, default=0.3)
    pa.add_argument("--seed",   type=int,   default=None,
                    help="Base seed for runs.")
    pa.add_argument("--runs",   type=int,   default=1,
                    help="Number of independent runs with different seeds.")
    pa.add_argument("--cores",  type=int,   default=1)
    pa.add_argument("--outdir", type=pathlib.Path,
                    default=pathlib.Path("deap_return_conn_3reg"))
    args = pa.parse_args()

    print("[INFO] NSGA‑II on 3‑regular graphs:")
    print("  • maximise mean return‑probability and algebraic connectivity")
    print(f"  • nodes={args.nodes}, gens={args.gens}, μ={args.mu}, λ={args.lam}")
    print(f"[INFO] Number of runs: {args.runs}")

    # Generate seeds for each run
    if args.seed is not None:
        seeds = [args.seed + i for i in range(args.runs)]
    else:
        seeds = [random.randint(0, 2**32 - 2) for _ in range(args.runs)]

    # Setup multiprocessing pool if requested
    pool = None
    if args.cores != 1:
        workers = mp.cpu_count() if args.cores < 0 else args.cores
        pool    = mp.Pool(workers)
        print(f"[INFO] using {workers} workers")

    # Get edges once for saving
    _, edges = make_toolbox(args.nodes, seeds[0])

    combined_points = []  # (mr, conn, graph_id, run)

    for run_idx, seed in enumerate(seeds):
        print(f"[INFO] Starting run {run_idx+1}/{args.runs} with seed={seed}")
        tb, _ = make_toolbox(args.nodes, seed)
        if pool is not None:
            tb.register("map", pool.map)

        # initial population & evaluation
        pop      = tb.population(n=args.mu)
        invalid  = [ind for ind in pop if not ind.fitness.valid]
        fits     = tb.map(tb.evaluate, invalid)
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        pop = tb.select(pop, len(pop))

        # run NSGA‑II
        run_hof = tools.ParetoFront()
        stats   = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda vals: tuple(np.mean([v[i] for v in vals]) for i in (0, 1)))
        stats.register("max", lambda vals: tuple(np.max([v[i] for v in vals]) for i in (0, 1)))

        algorithms.eaMuPlusLambda(
            pop, tb,
            mu=args.mu, lambda_=args.lam,
            cxpb=0.0, mutpb=args.mutpb,
            ngen=args.gens, halloffame=run_hof,
            stats=stats, verbose=True
        )

        # Save this run's output and collect points
        combined_points.extend(
            save_pareto_run(run_idx, run_hof, edges, args.nodes, args.outdir)
        )

    if pool is not None:
        pool.close(); pool.join()

    # Save global scatter and CSV of *all* points
    save_combined_scatter(combined_points, args.outdir)
    save_combined_csv(combined_points, args.outdir)
    print(f"\n[✓] Total points plotted: {len(combined_points)}")
    print(f"[✓] CSV saved to {args.outdir / 'combined_points.csv'}")

if __name__ == "__main__":
    main()

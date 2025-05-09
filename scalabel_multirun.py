#!/usr/bin/env python
# conn_cubic_evol_return_scalabel_multirun.py

"""
NSGA-II over 3-regular graphs, maximising
  (1) average return probability across all nodes (CTQW),
  (2) algebraic connectivity (λ₂ / n).
Supports multiple runs with different seeds and reconstructs the global Pareto front.
"""
from __future__ import annotations
import os, argparse, random, pathlib, multiprocessing as mp
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

# ────────────────── Quantum‐walk helpers ──────────────────
def ctqw_amplitudes(A: np.ndarray, s: int, times: np.ndarray) -> np.ndarray:
    w, V = eigh(A)
    # shape: (n, len(times))
    return V @ (V[s, :, None] * np.exp(-1j * np.outer(w, times)))

def mean_return_prob(A: np.ndarray, times: np.ndarray) -> float:
    n = A.shape[0]
    returns = []
    for s in range(n):
        amps = ctqw_amplitudes(A, s, times)
        # amplitude at start node s over time
        p_ret = np.abs(amps[s, :])**2
        returns.append(p_ret.mean())
    return float(np.mean(returns))

# ───────────── Secondary objective: algebraic connectivity ─────────────
def algebraic_conn_norm(G: nx.Graph, n: int) -> float:
    if not nx.is_connected(G):
        return 0.0
    L = nx.laplacian_matrix(G).astype(float).todense()
    λ2 = np.linalg.eigvalsh(L)[1]
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

# ───────────── Bi‐objective evaluation ─────────────
def evaluate(ind, edges, n, times):
    G = bits_to_graph(ind, edges, n)
    A = nx.to_numpy_array(G, dtype=float)

    # objective 1: mean return probability
    mr = mean_return_prob(A, times)

    # objective 2: algebraic connectivity
    conn = algebraic_conn_norm(G, n)

    return mr, conn

# ───────────── Build toolbox ─────────────
def make_toolbox(n, seed=None):
    rng = random.Random(seed)
    edges = list(combinations(range(n), 2))
    times = np.linspace(0, 20, 60)

    tb = base.Toolbox()

    # init: random 3-regular → bit‐vector
    def init_regular():
        G = nx.random_regular_graph(d=3, n=n, seed=rng)
        return creator.Individual(graph_to_bits(G, edges))
    tb.register("individual", init_regular)
    tb.register("population", tools.initRepeat, list, tb.individual)

    tb.register("evaluate", evaluate,
                edges=edges, n=n, times=times)

    # mutation: single double‐edge swap (preserves regularity)
    def mutate_regular(ind):
        G = bits_to_graph(ind, edges, n)
        nx.double_edge_swap(G, nswap=1, max_tries=100)
        ind[:] = graph_to_bits(G, edges)
        return (ind,)
    tb.register("mutate", mutate_regular)

    # no crossover
    tb.register("select", tools.selNSGA2)
    return tb, edges

# ───────────── Save helper ─────────────
def save_pareto(hof, edges, n, outdir):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Save individual graph visuals
    for i, ind in enumerate(hof):
        G = bits_to_graph(ind, edges, n)
        mr, conn = ind.fitness.values
        name = f"pareto_{i}_ret{mr:.3f}_conn{conn:.3f}"
        nx.write_graphml(G, outdir/f"{name}.graphml")
        plt.figure(figsize=(4,3))
        nx.draw_spring(G, node_size=80, with_labels=False)
        plt.title(f"⟨P_ret⟩={mr:.3f}, λ₂/n={conn:.3f}")
        plt.tight_layout()
        plt.savefig(outdir/f"{name}.png", dpi=150)
        plt.close()
    # Plot Pareto front: connectivity vs mean return with labels
    mrs   = [ind.fitness.values[0] for ind in hof]
    conns = [ind.fitness.values[1] for ind in hof]
    plt.figure(figsize=(5,4))
    plt.scatter(conns, mrs)
    for i, (conn_val, mr_val) in enumerate(zip(conns, mrs)):
        plt.annotate(
            str(i),
            xy=(conn_val, mr_val),
            xytext=(5, 5),
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=8
        )
    plt.xlabel("Algebraic connectivity (λ₂/n)")
    plt.ylabel("Mean return probability")
    plt.title("Pareto front: Connectivity vs Mean Return")
    plt.tight_layout()
    plt.savefig(outdir/"pareto_scatter_labeled.png", dpi=150)
    plt.close()

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

    print("[INFO] NSGA-II on 3-regular graphs:")
    print("  • maximise mean return-probability and algebraic connectivity")
    print(f"  • nodes={args.nodes}, gens={args.gens}, μ={args.mu}, λ={args.lam}")
    print(f"[INFO] Number of runs: {args.runs}")

    # Generate seeds for each run
    seeds = []
    if args.seed is not None:
        seeds = [args.seed + i for i in range(args.runs)]
    else:
        for _ in range(args.runs):
            seeds.append(random.randint(0, 2**32 - 2))

    # Prepare global Pareto front
    global_hof = tools.ParetoFront()

    # Setup multiprocessing pool if requested
    if args.cores != 1:
        workers = mp.cpu_count() if args.cores < 0 else args.cores
        pool = mp.Pool(workers)
        print(f"[INFO] using {workers} workers")

    # Get edges once for saving
    _, edges = make_toolbox(args.nodes, seeds[0])

    for run_idx, seed in enumerate(seeds):
        print(f"[INFO] Starting run {run_idx+1}/{args.runs} with seed={seed}")
        tb, _ = make_toolbox(args.nodes, seed)
        if args.cores != 1:
            tb.register("map", pool.map)

        # initial population & evaluation
        pop = tb.population(n=args.mu)
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits    = tb.map(tb.evaluate, invalid)
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        pop = tb.select(pop, len(pop))

        # run NSGA-II
        run_hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda vals: tuple(np.mean([v[i] for v in vals]) for i in (0,1)))
        stats.register("max", lambda vals: tuple(np.max([v[i] for v in vals]) for i in (0,1)))

        algorithms.eaMuPlusLambda(
            pop, tb,
            mu=args.mu, lambda_=args.lam,
            cxpb=0.0, mutpb=args.mutpb,
            ngen=args.gens, halloffame=run_hof,
            stats=stats, verbose=True
        )

        # update global Pareto front
        global_hof.update(run_hof)

    if args.cores != 1:
        pool.close(); pool.join()

    # Save combined Pareto front
    save_pareto(global_hof, edges, args.nodes, args.outdir)
    print(f"\n[✓] Global Pareto front size: {len(global_hof)}")

if __name__ == "__main__":
    main()

#python scalabel_multirun.py   --nodes 32   --gens 800   --mu 120   --lam 360   --mutpb 0.3   --seed 212   --runs 30   --cores 8   --outdir multirun_32_30_runs

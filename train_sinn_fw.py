#!/usr/bin/env python3
"""
Train a Structure-Imposing Neural Network (SINN) on Fisher-Wright adaptive-evolution
trajectories following Corrao *et al.* (2024).

2025-06-09 fix:
    • Handle global-slowdown resampling properly so the training trajectories
      (potentially < --seq-len) and SINN predictions are always compared on
      equal footing.

2025-06-10 additions:
    • Added higher-order ACF (β²) and 1-Wasserstein PDF loss terms.
    • Logged W₂ distance at the end of training.
    • Collected training/validation loss history.
    • Plotted error curves, trajectory overlay, dual-panel ACFs, PDF match, and
      a QQ-plot for distribution-level inspection.
"""
from __future__ import annotations

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from toolbox import SINN, StatLoss, make_loss

################################################################################
# CLI
################################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SINN on competing-pressure Fisher–Wright model (Corrao et al. 2024)"
    )
    a = p.add_argument

    # Simulation basics
    a("--popsize",  type=int,   default=10_000)
    a("--seq-len",  type=int,   default=400,
      help="requested generations; final T may be smaller after slowdown")
    a("--batch",    type=int,   default=400)
    a("--alpha-bar",type=float, default=0.3)
    a("--beta-bar", type=float, default=0.5)

    # Mutation / fitness parameters
    a("--U-alpha",  type=float, default=5e-7)
    a("--U-beta",   type=float, default=5e-7)
    a("--s-alpha",  type=float, default=0.02)
    a("--s-beta",   type=float, default=0.02)
    a("--a",        type=float, default=50.0, help="mutagenesis boost for U_α")
    a("--b",        type=float, default=50.0, help="mutagenesis boost for U_β")

    # Training
    a("--steps",    type=int,   default=20_000)
    a("--device",   choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()

################################################################################
# Fisher–Wright simulation
################################################################################

def simulate_two_trait(*,
                       N:int,
                       generations:int,
                       alpha_bar:float, beta_bar:float,
                       U_alpha:float, U_beta:float,
                       s_alpha:float, s_beta:float,
                       a:float, b:float) -> np.ndarray:
    """Return trajectory of mean β-mutation counts over `generations`."""
    s_alpha_eff = alpha_bar * s_alpha
    s_beta_eff  = beta_bar  * s_beta
    U_alpha_eff = U_alpha * (1 + a * alpha_bar)
    U_beta_eff  = U_beta  * (1 + b * alpha_bar)

    p_alpha, p_beta = U_alpha_eff, U_beta_eff
    k_alpha = np.zeros(N, dtype=np.int16)
    k_beta  = np.zeros(N, dtype=np.int16)

    traj = np.empty(generations, dtype=np.float32)
    rng  = np.random.default_rng()

    for t in range(generations):
        traj[t] = k_beta.mean()
        fit = 1.0 + k_alpha * s_alpha_eff + k_beta * s_beta_eff
        fit /= fit.mean()
        idx = rng.choice(N, N, replace=True, p=fit / fit.sum())
        k_alpha = k_alpha[idx].copy()
        k_beta  = k_beta[idx].copy()
        k_alpha += rng.binomial(1, p_alpha, N)
        k_beta  += rng.binomial(1, p_beta,  N)

    # Global-slowdown resampling
    slowdown = (1 - alpha_bar) * (1 - beta_bar)
    if slowdown < 1.0:
        new_len = max(1, int(generations * slowdown))
        idx = np.linspace(0, generations-1, new_len, dtype=int)
        traj = traj[idx]

    return traj

################################################################################
# Dataset builder
################################################################################

def build_dataset(args) -> tuple[torch.Tensor, torch.Tensor, int]:
    trajs = [
        simulate_two_trait(
            N=args.popsize,
            generations=args.seq_len,
            alpha_bar=args.alpha_bar, beta_bar=args.beta_bar,
            U_alpha=args.U_alpha, U_beta=args.U_beta,
            s_alpha=args.s_alpha, s_beta=args.s_beta,
            a=args.a, b=args.b
        )
        for _ in range(args.batch)
    ]
    T = trajs[0].shape[0]                          # actual sequence length
    trajs = np.stack(trajs, axis=1).astype(np.float32)  # (T,B)
    target     = torch.from_numpy(trajs[:, :, None])    # (T,B,1)
    val_noise  = torch.randn(2*T, args.batch, 1)        # validation input
    return target, val_noise, T

################################################################################
# Training loop + diagnostics
################################################################################

def train(args):
    # ---------------------------------------------------------------- devices
    dev = torch.device(
        "cuda" if (args.device == "cuda" or
                   (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    # ---------------------------------------------------------------- data
    target, val_noise, T = build_dataset(args)
    target, val_noise = target.to(dev), val_noise.to(dev)

    # ---------------------------------------------------------------- losses
    loss_acf      = make_loss("acf[fft]", target, lags=T, device=dev)
    loss_pdf      = make_loss("pdf",      target, lower=0, upper=5, n=T,
                              device=dev)
    loss_acf_sq   = make_loss("acf[fft]", target**2, lags=T, device=dev)
    loss_pdf_w2   = make_loss("pdf", target, lower=0, upper=5, n=T,
                              device=dev, dist="wasserstein")

    sinn = SINN(1, 25, 2, 1).to(dev)
    opt  = optim.Adam(sinn.parameters(), lr=1e-3)

    # history for plots
    hist = {"step": [], "train": [], "val": []}

    # ---------------------------------------------------------------- train
    for step in range(args.steps):
        sinn.train(); opt.zero_grad()

        noise = torch.randn(2*T, args.batch, 1, device=dev)
        pred, _ = sinn(noise)
        pred = pred[-T:]                                     # align length

        loss = (loss_acf(pred) +
                loss_pdf(pred) +
                0.1 * loss_acf_sq(pred) +
                0.1 * loss_pdf_w2(pred))

        loss.backward(); opt.step()

        # ------------------------------------------------------ validation log
        if step % 50 == 0 or step == args.steps-1:
            sinn.eval()
            with torch.no_grad():
                pv, _ = sinn(val_noise)
                pv = pv[-T:]
                val = (loss_acf(pv) + loss_pdf(pv) +
                       0.1 * loss_acf_sq(pv) +
                       0.1 * loss_pdf_w2(pv))

            print(f"[{step:5d}] train={loss.item():.4f}  val={val.item():.4f}")
            hist["step"].append(step)
            hist["train"].append(loss.item())
            hist["val"].append(val.item())

            if loss.item() < 0.01 and opt.param_groups[0]["lr"] > 1e-4:
                opt.param_groups[0]["lr"] *= 0.3

    # ---------------------------------------------------------------- metrics
    sinn.eval()
    with torch.no_grad():
        pf, _ = sinn(val_noise)
        pf = pf[-T:]

    acf_pred     = StatLoss.acf(pf,      method="bruteforce").mean(1).cpu()
    acf_true     = StatLoss.acf(target,  method="bruteforce").mean(1).cpu()
    acf_sq_pred  = StatLoss.acf(pf**2,   method="bruteforce").mean(1).cpu()
    acf_sq_true  = StatLoss.acf(target**2, method="bruteforce").mean(1).cpu()

    kde_pred   = StatLoss.gauss_kde(pf,     0, 5, 200).cpu()
    kde_target = StatLoss.gauss_kde(target, 0, 5, 200).cpu()
    w2 = torch.cdist(kde_pred[None], kde_target[None], p=2).item()
    print(f"Wasserstein-2 distance between PDFs: {w2:.4e}")

    # ---------------------------------------------------------------- plots
    t = np.arange(T)
    x = np.linspace(0, 5, 200)
    plt.style.use("ggplot")

    # 1) Loss history
    plt.figure(figsize=(4.5, 3))
    plt.loglog(hist["step"], hist["train"], label="train")
    plt.loglog(hist["step"], hist["val"],   "--", label="val")
    plt.title("Loss history")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend()

    # 2) Trajectory overlay
    plt.figure(figsize=(4.5, 3))
    plt.plot(t, target[:, 0, 0].cpu(), label="target")
    plt.plot(t, pf[:, 0, 0].cpu(), "--", label="SINN")
    plt.title("Example trajectory")
    plt.xlabel("generation"); plt.ylabel("β-mutations"); plt.legend()

    # 3) ACFs (β and β²)
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 2, 1)
    plt.plot(t, acf_true, label="target")
    plt.plot(t, acf_pred, "--", label="SINN")
    plt.title("ACF"); plt.xlabel("lag"); plt.ylabel("C(t)/C(0)"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, acf_sq_true, label="target")
    plt.plot(t, acf_sq_pred, "--", label="SINN")
    plt.title("ACF of β²")
    plt.xlabel("lag"); plt.ylabel("C₂(t)/C₂(0)"); plt.legend()

    # 4) PDFs
    plt.figure(figsize=(4.5, 3))
    plt.plot(x, kde_target, label="target")
    plt.plot(x, kde_pred, "--", label="SINN")
    plt.title("Stationary PDF")
    plt.xlabel("mean β-mutations"); plt.ylabel("ρ(x)"); plt.legend()

    # 5) QQ-plot
    target_flat = target.view(-1).cpu().numpy()
    pred_flat   = pf.view(-1).cpu().numpy()
    q = np.linspace(0.01, 0.99, 99)
    qt = np.quantile(target_flat, q)
    qp = np.quantile(pred_flat,   q)
    plt.figure(figsize=(4, 4))
    plt.plot(qt, qp, ".", ms=4)
    plt.plot(qt, qt, "k--")
    plt.title("QQ-plot")
    plt.xlabel("target quantile"); plt.ylabel("SINN quantile")

    plt.tight_layout()
    plt.show()

################################################################################
if __name__ == "__main__":
    train(parse_args())

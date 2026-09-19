"""
make_figures.py -- regenerates the four figures in main.tex from the cached runs.

Run it in the same directory as X.npy / W.npy / Y.npy, after stage.py has finished:

    CACHE=/content/drive/MyDrive/uplift_cache python3 make_figures.py

Produces fig1_pipeline.pdf, fig2_eda.pdf, fig3_qini.pdf, fig4_upliftk.pdf.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from run_pipeline import qini_curve, qini_stats, smd, uplift_at_k

CACHE = os.environ.get("CACHE", "cache")
N_BOOT_BAND = int(os.environ.get("N_BOOT_BAND", 200))
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
COL = {"T": "#1f77b4", "X": "#ff7f0e", "CF": "#2ca02c", "ENS": "#9467bd"}
NAME = {"T": "T-Learner", "X": "X-Learner", "CF": "Causal Forest", "ENS": "Ensemble"}

R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CACHE, "result_s*.json")))]
print(f"{len(R)} seeds from {CACHE}")


# --------------------------------------------------------------- Fig 1
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(3.4, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")
    bands = [(12.4, 2.4, "Data preparation", "#eaf3fb"),
             (8.9, 2.6, "Causal / uplift modelling", "#f2ecfa"),
             (5.6, 2.4, "Ensemble and evaluation", "#eafaf0"),
             (3.0, 1.8, "Decision output", "#fdf3e6")]
    for y, h, lab, c in bands:
        ax.add_patch(FancyBboxPatch((0.2, y - h + 0.2), 9.6, h,
                                    boxstyle="round,pad=0.12", fc=c, ec="#c8c8c8", lw=0.6))
        ax.text(0.5, y + 0.02, lab, fontsize=7.5, weight="bold", va="top")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                    fc=fc, ec="white", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=6.4, color="white", linespacing=1.25)

    box(0.6, 11.0, 2.5, 1.1, "MegaFon\n600k x 50", "#2e7d5b")
    box(3.75, 11.0, 2.5, 1.1, "Balance check\nSMD, 50 covars", "#d98324")
    box(6.9, 11.0, 2.5, 1.1, "Stratified split\n70/30, per seed", "#2f7d8c")
    box(0.6, 7.5, 2.5, 1.1, "T-Learner\nXGBoost", "#b5453b")
    box(3.75, 7.5, 2.5, 1.1, "X-Learner\nXGBoost", "#2f6fb5")
    box(6.9, 7.5, 2.5, 1.1, "Causal Forest\nEconML DML", "#7b4fa8")
    box(1.3, 4.3, 3.2, 1.1, "Ensemble\ntrain-only z-scores", "#c2407a")
    box(5.3, 4.3, 3.2, 1.1, "Evaluation\nQini, AUUC, U@k, CI", "#2f7d8c")
    box(2.6, 1.6, 4.8, 1.1, "Targeting policy\nrank + do-not-treat segment", "#d98324")

    for a, b in [((4.9, 11.0), (4.9, 8.6)), ((4.9, 7.5), (4.9, 5.4)), ((4.9, 4.3), (4.9, 2.7))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                     color="#888888", lw=0.9))
    fig.savefig("fig1_pipeline.pdf")
    plt.close(fig)


# --------------------------------------------------------------- Fig 2
def fig_eda():
    X = np.load("X.npy"); W = np.load("W.npy").astype(int); Y = np.load("Y.npy").astype(int)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))

    n1, n0 = (W == 1).sum(), (W == 0).sum()
    ax[0].bar(["treatment", "control"], [n1, n0], color=["#2f6fb5", "#d98324"], width=.55)
    ax[0].set_ylabel("subscribers")
    ax[0].set_title("(a) Arm balance")
    for i, (v, n) in enumerate(zip([n1, n0], [n1, n0])):
        ax[0].text(i, v, f"{100*n/len(W):.2f}%", ha="center", va="bottom", fontsize=6.5)

    p = [Y[W == 0].mean(), Y[W == 1].mean()]
    se = [np.sqrt(q * (1 - q) / m) for q, m in zip(p, [n0, n1])]
    ax[1].bar(["control", "treatment"], [100 * v for v in p],
              yerr=[196 * v for v in se], capsize=3,
              color=["#d98324", "#2f6fb5"], width=.55)
    ax[1].set_ylabel("conversion rate (%)")
    ax[1].set_title("(b) Conversion by arm")
    ax[1].text(0.5, max(p) * 100 * 1.06,
               f"ATE = {100*(p[1]-p[0]):.2f} pp", ha="center", fontsize=6.5)

    d = np.array([smd(X[:, i], W) for i in range(X.shape[1])])
    ax[2].scatter(d, np.arange(1, 51), s=5, color="#2e7d5b")
    ax[2].axvline(0, color="#999999", lw=.6)
    for t in (-0.1, 0.1):
        ax[2].axvline(t, color="#b5453b", ls="--", lw=.7)
    ax[2].set_xlim(-0.12, 0.12)
    ax[2].set_xlabel("standardised mean difference")
    ax[2].set_ylabel("covariate index")
    ax[2].set_title(f"(c) Balance, max |SMD| = {np.abs(d).max():.4f}")

    fig.tight_layout()
    fig.savefig("fig2_eda.pdf")
    plt.close(fig)


# --------------------------------------------------------------- Fig 3
def fig_qini():
    """Mean Qini curve across all seeds, with the across-seed 95% band."""
    seeds = sorted(int(os.path.basename(f).split("_")[0][1:])
                   for f in glob.glob(os.path.join(CACHE, "s*_tx.npz")))
    grid = np.linspace(0, 1, 201)
    curves = {k: [] for k in ("T", "X", "CF", "ENS")}
    ends = []
    for sd in seeds:
        d = np.load(os.path.join(CACHE, f"s{sd}_tx.npz"))
        Yte, Wte = d["Yte"], d["Wte"]
        cfs = sorted(glob.glob(os.path.join(CACHE, f"s{sd}_cf*.npz")))
        cf_te = np.mean([np.load(f)["te"] for f in cfs], axis=0)
        cf_tr = np.mean([np.load(f)["tr"] for f in cfs], axis=0)
        sc = {"T": d["t_te"], "X": d["x_te"], "CF": cf_te}
        tr = {"T": d["t_tr"], "X": d["x_tr"], "CF": cf_tr}
        sc["ENS"] = np.mean([(sc[k] - tr[k].mean()) / tr[k].std()
                             for k in ("T", "X", "CF")], axis=0)
        for k in curves:
            c = qini_curve(Yte, Wte, sc[k])
            curves[k].append(np.interp(grid, np.arange(len(c)) / (len(c) - 1), c))
        ends.append(qini_curve(Yte, Wte, sc["T"])[-1])

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for k in ("T", "X", "CF", "ENS"):
        A = np.array(curves[k])
        m = A.mean(0)
        q = np.mean([r["models"][k]["qini_raw"] for r in R])
        if len(seeds) > 1:
            lo, hi = np.percentile(A, [2.5, 97.5], axis=0)
            ax.fill_between(grid, lo, hi, color=COL[k], alpha=.15, lw=0)
        ax.plot(grid, m, color=COL[k], lw=1.2, label=f"{NAME[k]} ({q:.1f})")
    ax.plot([0, 1], [0, np.mean(ends)], "--", color="#888888", lw=.9,
            label="Random targeting")
    ax.set_xlabel("proportion of population targeted")
    ax.set_ylabel("cumulative incremental conversions")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig("fig3_qini.pdf")
    plt.close(fig)


# --------------------------------------------------------------- Fig 4
def fig_upliftk():
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ks = ["u10", "u20", "u30"]
    xs = np.arange(3)
    w = 0.2
    for j, k in enumerate(["T", "X", "CF", "ENS"]):
        m = [np.mean([r["models"][k][kk] for r in R]) for kk in ks]
        e = [np.std([r["models"][k][kk] for r in R]) for kk in ks]
        ax.bar(xs + (j - 1.5) * w, m, w, yerr=e, capsize=2,
               color=COL[k], label=NAME[k], error_kw=dict(lw=.7))
    ax.set_xticks(xs)
    ax.set_xticklabels(["Uplift@10%", "Uplift@20%", "Uplift@30%"])
    ax.set_ylabel("incremental conversion rate (pp)")
    ax.legend(frameon=False, ncol=2, fontsize=6.5)
    fig.tight_layout()
    fig.savefig("fig4_upliftk.pdf")
    plt.close(fig)


for f in (fig_pipeline, fig_eda, fig_qini, fig_upliftk):
    f()
    print("wrote", f.__name__)
print("\nDone. Copy the four PDFs next to main.tex.")

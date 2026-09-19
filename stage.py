"""
Resumable staged driver.  Each invocation executes exactly ONE pending step
and checkpoints to disk, so the whole experiment can be run under a short
per-command wall-clock limit.  On a normal workstation just set
FULL=1 to run everything in one process.

Steps per seed:
    tx          fit T-Learner and X-Learner, cache train/test uplift scores
    cf<k>       fit one chunk of CF_CHUNK causal trees, cache scores
    final       build the ensemble (train-only normalisation), compute all
                metrics, paired bootstrap, calibration, do-not-treat segment
"""
import json, os, sys, time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from run_pipeline import (XGB_OUTCOME, XGB_REG, XGB_NUIS, qini_stats,
                          uplift_at_k, smd, fit_t_learner, fit_x_learner)

N_SUB     = int(os.environ.get("N_SUB", 150_000))
SEEDS     = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
CF_TREES  = int(os.environ.get("CF_TREES", 100))    # EconML default
CF_CHUNK  = int(os.environ.get("CF_CHUNK", 50))     # trees per invocation
N_BOOT    = int(os.environ.get("N_BOOT", 1000))
CACHE     = os.environ.get("CACHE", "cache")
os.makedirs(CACHE, exist_ok=True)
NCHUNK    = CF_TREES // CF_CHUNK


def say(m):
    with open("progress.log", "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")
    print(m, flush=True)


def split_for(seed):
    X = np.load("X.npy"); W = np.load("W.npy").astype(int); Y = np.load("Y.npy").astype(int)
    rng = np.random.default_rng(seed)
    if N_SUB < len(X):
        strata = W * 2 + Y
        idx = np.concatenate([
            rng.choice(np.where(strata == s)[0],
                       int(round(N_SUB * (strata == s).mean())), replace=False)
            for s in np.unique(strata)])
        rng.shuffle(idx)
    else:
        idx = np.arange(len(X))
    Xs, Ws, Ys = X[idx], W[idx], Y[idx]
    itr, ite = train_test_split(np.arange(len(Xs)), test_size=0.30,
                                stratify=Ws * 2 + Ys, random_state=seed)
    return Xs[itr], Ws[itr], Ys[itr], Xs[ite], Ws[ite], Ys[ite]


def p(seed, name):
    return os.path.join(CACHE, f"s{seed}_{name}.npz")


def step_tx(seed):
    Xtr, Wtr, Ytr, Xte, Wte, Yte = split_for(seed)
    t = time.time()
    f_t, (m1, m0) = fit_t_learner(Xtr, Wtr, Ytr)
    t_tr, t_te = f_t(Xtr), f_t(Xte)
    tt = time.time() - t
    t = time.time()
    f_x, g = fit_x_learner(Xtr, Wtr, Ytr, m1, m0, seed)
    x_tr, x_te = f_x(Xtr), f_x(Xte)
    gp = g.predict_proba(Xte)[:, 1]
    tx = time.time() - t
    np.savez(p(seed, "tx"), t_tr=t_tr, t_te=t_te, x_tr=x_tr, x_te=x_te,
             Yte=Yte, Wte=Wte, prop=gp,
             prop_auc=roc_auc_score(Wte, gp), time_t=tt, time_x=tx)
    say(f"seed {seed} tx: T {tt:.0f}s  X {tx:.0f}s  prop mean {gp.mean():.4f} "
        f"AUC {roc_auc_score(Wte, gp):.4f}")


def step_cf(seed, k):
    from econml.dml import CausalForestDML
    Xtr, Wtr, Ytr, Xte, Wte, Yte = split_for(seed)
    t = time.time()
    cf = CausalForestDML(
        model_y=xgb.XGBRegressor(**XGB_NUIS),
        model_t=xgb.XGBClassifier(**XGB_NUIS, eval_metric="logloss"),
        discrete_treatment=True, n_estimators=CF_CHUNK, cv=2,
        random_state=1000 * seed + k, n_jobs=1)
    cf.fit(Ytr.astype(float), Wtr.astype(float), X=Xtr.astype(float), inference=None)
    np.savez(p(seed, f"cf{k}"),
             tr=cf.effect(Xtr.astype(float)).ravel(),
             te=cf.effect(Xte.astype(float)).ravel(),
             time=time.time() - t)
    say(f"seed {seed} cf chunk {k} ({CF_CHUNK} trees): {time.time()-t:.0f}s")


def step_final(seed):
    d = np.load(p(seed, "tx"))
    Yte, Wte = d["Yte"], d["Wte"]
    cf_tr = np.mean([np.load(p(seed, f"cf{k}"))["tr"] for k in range(NCHUNK)], axis=0)
    cf_te = np.mean([np.load(p(seed, f"cf{k}"))["te"] for k in range(NCHUNK)], axis=0)
    tr = {"T": d["t_tr"], "X": d["x_tr"], "CF": cf_tr}
    te = {"T": d["t_te"], "X": d["x_te"], "CF": cf_te}

    # ---- leakage-free ensemble: mu/sigma from TRAIN predictions only ----
    te["ENS"] = np.mean([(te[k] - tr[k].mean()) / (tr[k].std() or 1.0)
                         for k in ("T", "X", "CF")], axis=0)
    # ---- leaky variant, reported only to quantify the inflation ----------
    te["ENS_leaky"] = np.mean([(te[k] - te[k].mean()) / (te[k].std() or 1.0)
                               for k in ("T", "X", "CF")], axis=0)

    run = {"seed": seed, "n_test": int(len(Yte)),
           "propensity_mean": float(d["prop"].mean()),
           "propensity_auc": float(d["prop_auc"]),
           "time_t": float(d["time_t"]), "time_x": float(d["time_x"]),
           "time_cf": float(sum(np.load(p(seed, f"cf{k}"))["time"] for k in range(NCHUNK))),
           "models": {}}
    for k, s in te.items():
        r = qini_stats(Yte, Wte, s)
        for kk in (10, 20, 30):
            r[f"u{kk}"] = uplift_at_k(Yte, Wte, s, kk) * 100  # percentage points
        run["models"][k] = r

    # ---- paired bootstrap vs T-Learner ----------------------------------
    t0 = time.time()
    rng = np.random.default_rng(7000 + seed)
    acc = {k: {"d_qini_raw": [], "d_qini_norm": [], "d_u10": []}
           for k in te if k != "T"}
    for _ in range(N_BOOT):
        b = rng.integers(0, len(Yte), len(Yte))
        yb, wb = Yte[b], Wte[b]
        base = qini_stats(yb, wb, te["T"][b])
        bu = uplift_at_k(yb, wb, te["T"][b], 10)
        for k in acc:
            s = qini_stats(yb, wb, te[k][b])
            acc[k]["d_qini_raw"].append(s["qini_raw"] - base["qini_raw"])
            acc[k]["d_qini_norm"].append(s["qini_norm"] - base["qini_norm"])
            acc[k]["d_u10"].append((uplift_at_k(yb, wb, te[k][b], 10) - bu) * 100)
    run["bootstrap_vs_T"] = {
        k: {m: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)),
                    hi=float(np.percentile(v, 97.5)),
                    p_gt0=float(np.mean(np.array(v) > 0)))
            for m, v in dd.items()} for k, dd in acc.items()}
    run["time_boot"] = time.time() - t0

    # ---- calibration deciles on the ensemble score ----------------------
    s = te["ENS"]; order = np.argsort(-s); dec = []
    for j in range(10):
        sl = order[j * len(s) // 10:(j + 1) * len(s) // 10]
        yy, ww = Yte[sl], Wte[sl]
        dec.append(dict(decile=j + 1, n=int(len(sl)), n_treat=int(ww.sum()),
                        n_ctrl=int((1 - ww).sum()),
                        pred_z=float(s[sl].mean()),
                        observed_pp=float((yy[ww == 1].mean() - yy[ww == 0].mean()) * 100)))
    run["calibration_deciles"] = dec

    # ---- do-not-treat segment (negative T-Learner uplift) ---------------
    neg = te["T"] < 0
    run["do_not_treat"] = dict(
        fraction=float(neg.mean()),
        observed_pp=float((Yte[neg & (Wte == 1)].mean() - Yte[neg & (Wte == 0)].mean()) * 100)
        if (neg & (Wte == 1)).any() and (neg & (Wte == 0)).any() else None)

    json.dump(run, open(os.path.join(CACHE, f"result_s{seed}.json"), "w"), indent=1)
    say(f"seed {seed} FINAL done ({time.time()-t0:.0f}s bootstrap)")


def pending():
    for seed in SEEDS:
        if not os.path.exists(p(seed, "tx")):
            return ("tx", seed, None)
        for k in range(NCHUNK):
            if not os.path.exists(p(seed, f"cf{k}")):
                return ("cf", seed, k)
        if not os.path.exists(os.path.join(CACHE, f"result_s{seed}.json")):
            return ("final", seed, None)
    return None


if __name__ == "__main__":
    full = os.environ.get("FULL", "0") == "1"
    while True:
        job = pending()
        if job is None:
            say("ALL STEPS COMPLETE"); break
        kind, seed, k = job
        {"tx": lambda: step_tx(seed),
         "cf": lambda: step_cf(seed, k),
         "final": lambda: step_final(seed)}[kind]()
        if not full:
            break

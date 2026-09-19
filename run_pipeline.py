"""
Leakage-free uplift benchmark on the MegaFon Uplift Competition dataset.
Fixes vs. the original manuscript:
  * ensemble z-score constants estimated on TRAIN only (no test-set leakage)
  * XGBoost used for the X-Learner propensity model (uniform base learner)
  * multiple seeds + paired bootstrap CIs
  * normalized Qini and AUUC reported alongside raw Qini
  * SMD covariate balance, propensity AUC, CATE calibration, do-not-treat segment
"""
import json, os, sys, time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

N_SUB      = int(os.environ.get("N_SUB", 150_000))
SEEDS      = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
RUN_CF     = os.environ.get("RUN_CF", "1") == "1"
CF_TREES   = int(os.environ.get("CF_TREES", 200))
OUT        = os.environ.get("OUT", "results.json")
N_BOOT     = int(os.environ.get("N_BOOT", 1000))

XGB_OUTCOME = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, n_jobs=1,
                   tree_method="hist", eval_metric="logloss")
XGB_REG     = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, n_jobs=1,
                   tree_method="hist")
XGB_NUIS    = dict(n_estimators=100, max_depth=8, max_leaves=100,
                   grow_policy="lossguide", tree_method="hist", n_jobs=1)


# ---------------------------------------------------------------- metrics
def qini_curve(y, w, score):
    """Cumulative incremental conversions as a function of targeted fraction."""
    order = np.argsort(-score, kind="mergesort")
    y, w = y[order], w[order]
    ct, cc = np.cumsum(w == 1), np.cumsum(w == 0)
    yt, yc = np.cumsum(y * (w == 1)), np.cumsum(y * (w == 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = yt - np.where(cc > 0, yc * ct / np.maximum(cc, 1), 0.0)
    curve = np.concatenate([[0.0], curve])
    return curve


def qini_stats(y, w, score):
    """Raw Qini area (incremental conversions), normalized Qini, AUUC."""
    curve = qini_curve(y, w, score)
    n = len(curve) - 1
    frac = np.arange(n + 1) / n
    end = curve[-1]
    rand = frac * end
    raw = np.trapezoid(curve - rand, frac)          # area vs random, abs units
    perfect = qini_curve(y, w, (y * (2 * w - 1)).astype(float))
    denom = np.trapezoid(perfect - rand, frac)
    norm = raw / denom if denom > 0 else np.nan
    auuc = np.trapezoid(curve, frac)                # area under curve itself
    return dict(qini_raw=float(raw), qini_norm=float(norm), auuc=float(auuc))


def uplift_at_k(y, w, score, k):
    order = np.argsort(-score, kind="mergesort")
    m = max(1, int(round(len(y) * k / 100)))
    ys, ws = y[order][:m], w[order][:m]
    if ws.sum() == 0 or (1 - ws).sum() == 0:
        return np.nan
    return float(ys[ws == 1].mean() - ys[ws == 0].mean())


def smd(x, w):
    a, b = x[w == 1], x[w == 0]
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


# ---------------------------------------------------------------- learners
def fit_t_learner(Xtr, Wtr, Ytr):
    m1 = xgb.XGBClassifier(**XGB_OUTCOME).fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
    m0 = xgb.XGBClassifier(**XGB_OUTCOME).fit(Xtr[Wtr == 0], Ytr[Wtr == 0])
    return lambda X: m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1], (m1, m0)


def fit_x_learner(Xtr, Wtr, Ytr, m1, m0, seed):
    d1 = Ytr[Wtr == 1] - m0.predict_proba(Xtr[Wtr == 1])[:, 1]
    d0 = m1.predict_proba(Xtr[Wtr == 0])[:, 1] - Ytr[Wtr == 0]
    t1 = xgb.XGBRegressor(**XGB_REG).fit(Xtr[Wtr == 1], d1)
    t0 = xgb.XGBRegressor(**XGB_REG).fit(Xtr[Wtr == 0], d0)
    # propensity: XGBoost (not random forest) -> uniform base learner
    g = xgb.XGBClassifier(**XGB_OUTCOME).fit(Xtr, Wtr)
    def pred(X):
        gp = g.predict_proba(X)[:, 1]
        return gp * t0.predict(X) + (1 - gp) * t1.predict(X)
    return pred, g


def fit_causal_forest(Xtr, Wtr, Ytr, seed):
    from econml.dml import CausalForestDML
    cf = CausalForestDML(
        model_y=xgb.XGBRegressor(**XGB_NUIS),
        model_t=xgb.XGBClassifier(**XGB_NUIS, eval_metric="logloss"),
        discrete_treatment=True, n_estimators=CF_TREES, cv=2,
        random_state=seed, n_jobs=1)
    cf.fit(Ytr.astype(float), Wtr.astype(float), X=Xtr.astype(float), inference=None)
    return lambda X: cf.effect(X.astype(float)).ravel()


# ---------------------------------------------------------------- main
def main():
    X = np.load("X.npy"); W = np.load("W.npy").astype(int); Y = np.load("Y.npy").astype(int)
    log = open("progress.log", "a", buffering=1)

    def say(m):
        log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n"); print(m, flush=True)

    # full-population covariate balance (computed once, on all 600k)
    balance = {f"X_{i+1}": smd(X[:, i], W) for i in range(X.shape[1])}
    say(f"max |SMD| over 50 covariates = {max(abs(v) for v in balance.values()):.4f}")

    all_runs = []
    for seed in SEEDS:
        t_seed = time.time()
        rng = np.random.default_rng(seed)
        # stratified subsample on (W, Y)
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
        strat = Ws * 2 + Ys
        itr, ite = train_test_split(np.arange(len(Xs)), test_size=0.30,
                                    stratify=strat, random_state=seed)
        Xtr, Wtr, Ytr = Xs[itr], Ws[itr], Ys[itr]
        Xte, Wte, Yte = Xs[ite], Ws[ite], Ys[ite]
        say(f"seed {seed}: train {len(itr)} test {len(ite)}")

        preds_tr, preds_te = {}, {}
        t0 = time.time()
        f_t, (m1, m0) = fit_t_learner(Xtr, Wtr, Ytr)
        preds_tr["T"], preds_te["T"] = f_t(Xtr), f_t(Xte)
        say(f"  T-Learner {time.time()-t0:.0f}s")

        t0 = time.time()
        f_x, g = fit_x_learner(Xtr, Wtr, Ytr, m1, m0, seed)
        preds_tr["X"], preds_te["X"] = f_x(Xtr), f_x(Xte)
        prop_auc = roc_auc_score(Wte, g.predict_proba(Xte)[:, 1])
        prop_mean = float(g.predict_proba(Xte)[:, 1].mean())
        say(f"  X-Learner {time.time()-t0:.0f}s  propensity mean={prop_mean:.4f} AUC={prop_auc:.4f}")

        if RUN_CF:
            t0 = time.time()
            f_c = fit_causal_forest(Xtr, Wtr, Ytr, seed)
            preds_tr["CF"], preds_te["CF"] = f_c(Xtr), f_c(Xte)
            say(f"  Causal Forest {time.time()-t0:.0f}s")

        # ---- leakage-free ensemble: mu/sigma from TRAIN predictions only ----
        keys = [k for k in ("T", "X", "CF") if k in preds_te]
        z_te = []
        for k in keys:
            mu, sd = preds_tr[k].mean(), preds_tr[k].std()
            z_te.append((preds_te[k] - mu) / (sd if sd > 0 else 1.0))
        preds_te["ENS"] = np.mean(z_te, axis=0)
        # leaky variant, for the ablation table only
        z_leak = [(preds_te[k] - preds_te[k].mean()) / preds_te[k].std() for k in keys]
        preds_te["ENS_leaky"] = np.mean(z_leak, axis=0)

        run = dict(seed=seed, n_train=len(itr), n_test=len(ite),
                   propensity_mean=prop_mean, propensity_auc=float(prop_auc),
                   models={})
        for k, s in preds_te.items():
            r = qini_stats(Yte, Wte, s)
            for kk in (10, 20, 30):
                r[f"u{kk}"] = uplift_at_k(Yte, Wte, s, kk)
            r["frac_negative"] = float((s < np.median(s) * 0 + 0).mean()) if k != "ENS" else float((s < 0).mean())
            run["models"][k] = r

        # paired bootstrap of Delta(metric) vs T-Learner on this seed's test set
        boots = {k: {"d_qini_raw": [], "d_qini_norm": [], "d_u10": []} for k in preds_te if k != "T"}
        bidx_rng = np.random.default_rng(1000 + seed)
        for _ in range(N_BOOT):
            b = bidx_rng.integers(0, len(Yte), len(Yte))
            yb, wb = Yte[b], Wte[b]
            if wb.sum() == 0 or (1 - wb).sum() == 0:
                continue
            base = qini_stats(yb, wb, preds_te["T"][b])
            base_u = uplift_at_k(yb, wb, preds_te["T"][b], 10)
            for k in boots:
                s = qini_stats(yb, wb, preds_te[k][b])
                boots[k]["d_qini_raw"].append(s["qini_raw"] - base["qini_raw"])
                boots[k]["d_qini_norm"].append(s["qini_norm"] - base["qini_norm"])
                boots[k]["d_u10"].append(uplift_at_k(yb, wb, preds_te[k][b], 10) - base_u)
        run["bootstrap_vs_T"] = {
            k: {m: dict(mean=float(np.mean(v)),
                        lo=float(np.percentile(v, 2.5)),
                        hi=float(np.percentile(v, 97.5)),
                        p_gt0=float(np.mean(np.array(v) > 0)))
                for m, v in d.items()}
            for k, d in boots.items()}

        # CATE calibration deciles + do-not-treat segment (ensemble score)
        s = preds_te["ENS"]
        order = np.argsort(-s)
        dec = []
        for j in range(10):
            sl = order[j * len(s) // 10:(j + 1) * len(s) // 10]
            yy, ww = Yte[sl], Wte[sl]
            dec.append(dict(decile=j + 1, n=len(sl),
                            pred=float(s[sl].mean()),
                            obs=float(yy[ww == 1].mean() - yy[ww == 0].mean()),
                            n_treat=int(ww.sum()), n_ctrl=int((1 - ww).sum())))
        run["calibration_deciles"] = dec
        # raw-scale do-not-treat: fraction with negative T-Learner uplift
        neg = preds_te["T"] < 0
        run["do_not_treat"] = dict(
            fraction=float(neg.mean()),
            observed_uplift=float(Yte[neg & (Wte == 1)].mean() - Yte[neg & (Wte == 0)].mean())
            if (neg & (Wte == 1)).any() and (neg & (Wte == 0)).any() else None)

        all_runs.append(run)
        json.dump(dict(balance_smd=balance, runs=all_runs), open(OUT, "w"), indent=1)
        say(f"seed {seed} done in {time.time()-t_seed:.0f}s -> {OUT}")

    say("ALL DONE")


if __name__ == "__main__":
    main()

# Uplift benchmark on the MegaFon dataset

Code for "Causal Machine Learning for Retention-Campaign Targeting: A Controlled
Benchmark of Heterogeneous Treatment-Effect Estimators in Telecommunications".

## Setup

    pip install -r requirements.txt

    curl -L -o megafon.csv.gz \
      https://github.com/jszymon/uplift_sklearn_data/releases/download/Megafon/megafon_dataset.csv.gz

    python3 -c "
    import pandas as pd, numpy as np
    df = pd.read_csv('megafon.csv.gz')
    np.save('X.npy', df[[f'X_{i}' for i in range(1,51)]].values.astype(np.float32))
    np.save('W.npy', (df.treatment_group=='treatment').values.astype(np.int8))
    np.save('Y.npy', df.conversion.values.astype(np.int8))"

## Reproduce

Multi-seed protocol (seeds 0-4, 150k stratified subsample):

    SEEDS=0,1,2,3,4 CF_TREES=200 CF_CHUNK=200 N_BOOT=1000 FULL=1 python3 stage.py

Full-corpus confirmation (600k, seed 0):

    N_SUB=600000 SEEDS=0 CF_TREES=200 CF_CHUNK=200 N_BOOT=1000 \
      CACHE=cache_full FULL=1 python3 stage.py

Figures:

    python3 make_figures.py

## Contents

| File | Purpose |
|---|---|
| `run_pipeline.py` | Estimators and metrics (Qini, AUUC, Uplift@k, SMD) |
| `stage.py` | Resumable driver, leakage-free ensemble, paired bootstrap |
| `make_figures.py` | Regenerates all four paper figures |
| `shap_uplift_importance.csv` | Full 50-covariate SHAP ranking on the uplift score |
| `results/` | Cached per-seed results reported in the paper |

Hardware: Intel Xeon @ 2.20 GHz, 2 cores, 12 GB RAM (Google Colab).
Seeds: {0,1,2,3,4} for the multi-seed protocol, seed 0 for the full-corpus run.

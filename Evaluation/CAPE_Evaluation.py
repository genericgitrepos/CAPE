import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict, LeaveOneGroupOut
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


CSV_PATH = "CAPE_DATASET.csv"
TARGET_KEYWORDS = ["t_conv", "tconv", "epochs", "t*conv"]
RANDOM_STATE = 42


RUN_SCALING_LAW = True
RUN_LCE_EXP = True
RUN_CAPE_PROBE_ONLY = True


def _lower_map(cols: List[str]) -> Dict[str, str]:
    out = {}
    for c in cols:
        lc = c.strip().lower()
        if lc not in out:
            out[lc] = c
    return out

def _find_col(lmap: Dict[str, str], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in lmap:
            return lmap[k]
    for want in keys:
        for have in lmap.keys():
            if want.replace("_", "").replace("-", "") == have.replace("_", "").replace("-", ""):
                return lmap[have]
    return None

def _guess_required_cols(df: pd.DataFrame) -> Dict[str, str]:
    lmap = _lower_map(list(df.columns))
    col_model   = _find_col(lmap, ["model", "arch", "architecture"])
    col_dataset = _find_col(lmap, ["dataset", "data", "ds"])
    col_lr      = _find_col(lmap, ["lr", "learning_rate", "eta", "loglr", "log_lr"])
    col_bs      = _find_col(lmap, ["batch", "batch_size", "bs", "logb", "log_b"])
    col_opt     = _find_col(lmap, ["optimizer", "optim", "opt"])

    col_logp    = _find_col(lmap, ["logp", "param_count_log10", "log_params", "log10_params"])
    col_logg2   = _find_col(lmap, ["logg2", "gradient_norm_log10", "g2_x_lr", "g2"])
    col_logtau  = _find_col(lmap, ["logtau", "ntk_trace_proxy_log10", "tau"])
    col_logn    = _find_col(lmap, ["logn", "dataset_size_log10", "n_log10"])
    col_loglr   = _find_col(lmap, ["loglr", "log_lr", "learning_rate_log10"])
    col_logb    = _find_col(lmap, ["logb", "log_b", "batch_size_log10"])

    col_pat     = _find_col(lmap, ["patience_p", "patience"])
    col_mdelta  = _find_col(lmap, ["min_delta"])
    col_maxep   = _find_col(lmap, ["max_epochs"])

    col_valpfx  = _find_col(lmap, ["val_loss_prefix", "valprefix", "val_loss_json"])

    tgt = None
    for k in TARGET_KEYWORDS:
        found = _find_col(lmap, [k])
        if found:
            tgt = found
            break
    if tgt is None and "t_conv" in lmap:
        tgt = lmap["t_conv"]

    need = {
        "model": col_model,
        "dataset": col_dataset,
        "lr": col_lr,
        "batch": col_bs,
        "optimizer": col_opt,
        "logP": col_logp,
        "logG2": col_logg2,
        "logTau": col_logtau,
        "logN": col_logn,
        "logLR": col_loglr,
        "logB": col_logb,
        "patience": col_pat,
        "min_delta": col_mdelta,
        "max_epochs": col_maxep,
        "val_loss_prefix": col_valpfx,
        "target": tgt,
    }
    missing = [k for k, v in need.items() if v is None and k in ["model", "dataset", "target"]]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    return need

def _normalize_model_name(x: str) -> str:
    try:
        s = str(x).lower().replace("-", "").replace("_", "")
        return s
    except Exception:
        return str(x)

def _hide_model(model_name: str) -> bool:
    s = _normalize_model_name(model_name)
    return s == "vit" or s == "vgg16"

def print_table(df: pd.DataFrame, title: str):
    print("\n" + title)
    print("-" * max(70, len(title)))
    if df.empty:
        print("[EMPTY]")
        return
    df2 = df.copy()
    for c in df2.select_dtypes(include=[float]).columns:
        df2[c] = df2[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    print(df2.to_string(index=False))

def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape_pct(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def _pearsonr_safe(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    if y_true.size < 2:
        return np.nan
    s1 = float(np.std(y_true))
    s2 = float(np.std(y_pred))
    if s1 < 1e-12 or s2 < 1e-12:
        return np.nan
    cov = float(np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))))
    return cov / (s1 * s2)

def _spearmanr_safe(y_true, y_pred):
    y_true = pd.Series(np.asarray(y_true, float)).rank(method="average").to_numpy()
    y_pred = pd.Series(np.asarray(y_pred, float)).rank(method="average").to_numpy()
    return _pearsonr_safe(y_true, y_pred)

def _bre(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), 1.0)
    rel = np.abs(y_true - y_pred) / denom
    return float(np.mean(np.minimum(rel, 1.0)))

def full_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": _rmse(y_true, y_pred),
        "MAPE%": _mape_pct(y_true, y_pred),
        "PearsonR": _pearsonr_safe(y_true, y_pred),
        "SpearmanR": _spearmanr_safe(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "BRE": _bre(y_true, y_pred),
    }

def base_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": _rmse(y_true, y_pred),
        "PearsonR": _pearsonr_safe(y_true, y_pred),
    }

def leave_one_group_out_oof(pipe, X: pd.DataFrame, y: np.ndarray, groups: pd.Series) -> np.ndarray:
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(y, dtype=float)
    for tr, te in logo.split(X, y, groups=groups):
        pipe.fit(X.iloc[tr], y[tr])
        y_pred[te] = pipe.predict(X.iloc[te])
    return y_pred

def _parse_prefix(s: str) -> Optional[List[float]]:
    if pd.isna(s):
        return None
    try:
        arr = json.loads(s) if isinstance(s, str) else list(s)
        arr = [float(x) for x in arr if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return arr if len(arr) >= 3 else None
    except Exception:
        return None

def _early_stop_epoch(val_losses: List[float], patience: int, min_delta: float) -> int:
    best = float('inf')
    no_improve = 0
    for i, v in enumerate(val_losses, start=1):
        if v < best - min_delta:
            best = v
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            return i
    return len(val_losses)

def _fit_exponential(prefix: List[float]) -> Tuple[float, float, float]:
    t = np.arange(1, len(prefix) + 1, dtype=float)
    y = np.asarray(prefix, dtype=float)
    c = float(min(np.min(y[-2:]) if len(y) >= 2 else y[-1], np.min(y)) * 0.99)
    z = y - c
    z[z <= 1e-8] = 1e-8
    X = np.vstack([np.ones_like(t), -t]).T
    target = np.log(z)
    try:
        theta, *_ = np.linalg.lstsq(X, target, rcond=None)
        loga, b = theta[0], theta[1]
        a = float(np.exp(loga))
        b = float(b)
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError
        return a, b, c
    except Exception:
        return float(max(y) - min(y)), 0.05, float(min(y))

def _aggregate_median_safe(arr: List[float], default_val: float) -> float:
    arr = [x for x in arr if x is not None and np.isfinite(x)]
    return float(np.median(arr)) if len(arr) else float(default_val)

def learn_fold_hypers_exp(train_df: pd.DataFrame, col_valpfx: str) -> Dict[str, float]:
    b_exp, c_exp = [], []
    for _, row in train_df.iterrows():
        prefix = _parse_prefix(row.get(col_valpfx))
        if not prefix:
            continue
        _, b, c = _fit_exponential(prefix)
        b_exp.append(b); c_exp.append(c)
    exp_b = _aggregate_median_safe(b_exp, 0.05)
    exp_c = _aggregate_median_safe(c_exp, 0.0)
    return {"b": exp_b, "c": exp_c}

def _calibrate_a_exp(prefix: List[float], b: float, c: float) -> float:
    t = np.arange(1, len(prefix) + 1, dtype=float)
    y = np.asarray(prefix, dtype=float)
    w = np.exp(-b * t)
    num = float(np.sum((y - c) * w))
    den = float(np.sum(w * w)) + 1e-12
    return num / den

def _exp_predict(a: float, b: float, c: float, t: int) -> float:
    return float(a * math.exp(-b * t) + c)

def _predict_tconv_from_fixed_exp(prefix: List[float],
                                  patience: int,
                                  min_delta: float,
                                  max_epochs: int,
                                  hypers: Dict[str, float]) -> Optional[int]:
    if not prefix or patience is None or min_delta is None or max_epochs is None:
        return None
    k = len(prefix)
    if k < 3:
        return None
    try:
        losses = list(map(float, prefix))
        b, c = float(hypers["b"]), float(hypers["c"])
        a = _calibrate_a_exp(losses, b, c)
        for t in range(k + 1, max_epochs + 1):
            losses.append(_exp_predict(a, b, c, t))
        tconv = _early_stop_epoch(losses, int(patience), float(min_delta))
        return int(tconv)
    except Exception:
        return None

def kfold_indices(n: int, n_splits: int = 5, seed: int = 42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))

def logo_indices(groups: pd.Series):
    logo = LeaveOneGroupOut()
    return list(logo.split(np.arange(len(groups)), groups=groups))

df_raw = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
cols = _guess_required_cols(df_raw)
df = df_raw[~df_raw[cols["target"]].isna()].reset_index(drop=True)

if cols["model"] in df.columns:
    mask_hide = df[cols["model"]].apply(_hide_model)
    df = df[~mask_hide].reset_index(drop=True)

forbidden = {cols["target"], "T_80close", "T_90close", cols.get("val_loss_prefix", "val_loss_prefix")}
feature_cols = [c for c in df.columns if c not in forbidden]

num_cols, cat_cols = [], []
for c in feature_cols:
    (num_cols if pd.api.types.is_numeric_dtype(df[c]) else cat_cols).append(c)
const_num = [c for c in num_cols if df[c].nunique(dropna=True) <= 1]
num_cols = [c for c in num_cols if c not in const_num]

single_cat = [c for c in cat_cols if df[c].nunique(dropna=True) <= 1]
cat_cols = [c for c in cat_cols if c not in single_cat]
feature_cols = [c for c in feature_cols if c not in single_cat]

COL_MODEL   = cols["model"]
COL_DATASET = cols["dataset"]
COL_LR      = cols.get("lr")
COL_BS      = cols.get("batch")
COL_OPT     = cols.get("optimizer")
COL_PAT     = cols.get("patience")
COL_MINDEL  = cols.get("min_delta")
COL_MAXEP   = cols.get("max_epochs")
COL_VALPFX  = cols.get("val_loss_prefix")
COL_LOGP    = cols.get("logP")
COL_LOGN    = cols.get("logN")
COL_LOGG2   = cols.get("logG2")
COL_LOGTAU  = cols.get("logTau")

X_all = df[feature_cols].copy()
y = df[cols["target"]].astype(float).to_numpy()

print(f"[INFO] Loaded {len(df)} rows from {CSV_PATH}")
print(f"[INFO] Numeric features: {len(num_cols)} | Categorical: {len(cat_cols)} | Dropped-constant: {len(const_num)+len(single_cat)}")

REGRESSOR = RandomForestRegressor(n_estimators=500, n_jobs=1, random_state=RANDOM_STATE)

def build_feature_matrix(df_in: pd.DataFrame, feature_cols: List[str],
                         num_cols: List[str], cat_cols: List[str],
                         drop_cols: List[str]):
    use = [c for c in feature_cols if c not in drop_cols]
    num = [c for c in num_cols if c in use]
    cat = [c for c in cat_cols if c in use]
    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ],
        remainder="drop"
    )
    return use, preproc

use_cv,  preproc_cv   = build_feature_matrix(df, feature_cols, num_cols, cat_cols, drop_cols=[])
use_lodo,preproc_lodo = build_feature_matrix(df, feature_cols, num_cols, cat_cols,
                                             drop_cols=[COL_DATASET] if COL_DATASET else [])
use_lomo,preproc_lomo = build_feature_matrix(df, feature_cols, num_cols, cat_cols,
                                             drop_cols=[COL_MODEL] if COL_MODEL else [])

pipe_cv   = Pipeline([("prep", preproc_cv),   ("reg", REGRESSOR)])
pipe_lodo = Pipeline([("prep", preproc_lodo), ("reg", REGRESSOR)])
pipe_lomo = Pipeline([("prep", preproc_lomo), ("reg", REGRESSOR)])

X_cv   = df[use_cv].copy()
X_lodo = df[use_lodo].copy()
X_lomo = df[use_lomo].copy()

def run_cv(pipe, X, y) -> np.ndarray:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)

def run_lodo(pipe, X, y, df_) -> np.ndarray:
    if COL_DATASET not in df_.columns:
        raise KeyError(f"Column '{COL_DATASET}' is required for LODO.")
    return leave_one_group_out_oof(pipe, X, y, df_[COL_DATASET])

def run_lomo(pipe, X, y, df_) -> np.ndarray:
    if COL_MODEL not in df_.columns:
        raise KeyError(f"Column '{COL_MODEL}' is required for LOMO.")
    return leave_one_group_out_oof(pipe, X, y, df_[COL_MODEL])

all_overall_rows = []
all_pred_rows = []

y_cv = run_cv(pipe_cv, X_cv, y)
all_overall_rows.append({"Method": "CAPE", "Regime": "CV (5-fold)", **full_metrics(y, y_cv)})
df_cv = df.copy()
df_cv["true_epochs"] = y; df_cv["pred_epochs"] = y_cv
df_cv["Method"] = "CAPE"; df_cv["Regime"] = "CV (5-fold)"
all_pred_rows.append(df_cv)

y_lodo = run_lodo(pipe_lodo, X_lodo, y, df)
all_overall_rows.append({"Method": "CAPE", "Regime": "LODO", **full_metrics(y, y_lodo)})
df_lodo = df.copy()
df_lodo["true_epochs"] = y; df_lodo["pred_epochs"] = y_lodo
df_lodo["Method"] = "CAPE"; df_lodo["Regime"] = "LODO"
all_pred_rows.append(df_lodo)

y_lomo = run_lomo(pipe_lomo, X_lomo, y, df)
all_overall_rows.append({"Method": "CAPE", "Regime": "LOMO", **full_metrics(y, y_lomo)})
df_lomo = df.copy()
df_lomo["true_epochs"] = y; df_lomo["pred_epochs"] = y_lomo
df_lomo["Method"] = "CAPE"; df_lomo["Regime"] = "LOMO"
all_pred_rows.append(df_lomo)

if RUN_CAPE_PROBE_ONLY:
    probe_feats = []
    if (COL_LOGG2 is not None) and (COL_LOGG2 in df.columns):
        probe_feats.append(COL_LOGG2)
    if (COL_LOGTAU is not None) and (COL_LOGTAU in df.columns):
        probe_feats.append(COL_LOGTAU)

    if len(probe_feats) >= 1:
        X_probe = df[probe_feats].copy()
        preproc_probe = ColumnTransformer(
            transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                            ("scaler", StandardScaler())]), probe_feats)],
            remainder="drop"
        )
        pipe_probe = Pipeline([("prep", preproc_probe), ("reg", REGRESSOR)])

        y_cv_p = run_cv(pipe_probe, X_probe, y)
        all_overall_rows.append({"Method": "CAPE (probe-only)", "Regime": "CV (5-fold)", **base_metrics(y, y_cv_p)})
        dcvp = df.copy(); dcvp["true_epochs"] = y; dcvp["pred_epochs"] = y_cv_p
        dcvp["Method"] = "CAPE (probe-only)"; dcvp["Regime"] = "CV (5-fold)"
        all_pred_rows.append(dcvp)

        y_lodo_p = run_lodo(pipe_probe, X_probe, y, df)
        all_overall_rows.append({"Method": "CAPE (probe-only)", "Regime": "LODO", **base_metrics(y, y_lodo_p)})
        dlodop = df.copy(); dlodop["true_epochs"] = y; dlodop["pred_epochs"] = y_lodo_p
        dlodop["Method"] = "CAPE (probe-only)"; dlodop["Regime"] = "LODO"
        all_pred_rows.append(dlodop)

        y_lomo_p = run_lomo(pipe_probe, X_probe, y, df)
        all_overall_rows.append({"Method": "CAPE (probe-only)", "Regime": "LOMO", **base_metrics(y, y_lomo_p)})
        dlomop = df.copy(); dlomop["true_epochs"] = y; dlomop["pred_epochs"] = y_lomo_p
        dlomop["Method"] = "CAPE (probe-only)"; dlomop["Regime"] = "LOMO"
        all_pred_rows.append(dlomop)
    else:
        print("[INFO] Probe-only CAPE skipped: probe features (logG2/logTau) not found.")

def _meta_defaults(df_in: pd.DataFrame) -> Tuple[int, float, int]:
    default_patience = 7
    default_mindelta = 5e-4
    default_maxep = 200
    if COL_MAXEP and (COL_MAXEP in df_in.columns) and df_in[COL_MAXEP].notna().any():
        default_maxep = int(np.nanmax(df_in[COL_MAXEP].values))
    return default_patience, default_mindelta, default_maxep

def _get_meta(row, key_col: Optional[str], default_val):
    try:
        if key_col and (key_col in row) and not pd.isna(row[key_col]):
            return row[key_col]
    except Exception:
        pass
    return default_val

def _predict_lce_exp_cv(df_in: pd.DataFrame, regime: str) -> np.ndarray:
    n = len(df_in)
    y_oof = np.full(n, np.nan, dtype=float)

    if regime == "CV (5-fold)":
        splits = kfold_indices(n, n_splits=5, seed=RANDOM_STATE)
    elif regime == "LODO":
        splits = logo_indices(df_in[COL_DATASET])
    elif regime == "LOMO":
        splits = logo_indices(df_in[COL_MODEL])
    else:
        raise ValueError(f"Unknown regime: {regime}")

    for tr_idx, te_idx in splits:
        train_df = df_in.iloc[tr_idx]
        test_df  = df_in.iloc[te_idx]

        pat0, md0, me0 = _meta_defaults(train_df)

        hypers = learn_fold_hypers_exp(train_df, COL_VALPFX) if RUN_LCE_EXP else {}

        for i, row in test_df.iterrows():
            prefix = _parse_prefix(row.get(COL_VALPFX)) if COL_VALPFX else None
            if not prefix:
                continue
            patience  = int(_get_meta(row, COL_PAT,   pat0))
            min_delta = float(_get_meta(row, COL_MINDEL, md0))
            max_epochs = int(_get_meta(row, COL_MAXEP, me0))
            t_pred = _predict_tconv_from_fixed_exp(prefix, patience, min_delta, max_epochs, hypers)
            if t_pred is not None:
                y_oof[i] = float(t_pred)
    return y_oof

if RUN_LCE_EXP:
    for regime in ["CV (5-fold)", "LODO", "LOMO"]:
        y_lce = _predict_lce_exp_cv(df, regime)
        mask = np.isfinite(y) & np.isfinite(y_lce)
        if mask.any():
            all_overall_rows.append({"Method": "LCE (exp; prefix, CV-learned hypers)", "Regime": regime,
                                     **base_metrics(y[mask], y_lce[mask])})
            d = df.loc[mask].copy()
            d["true_epochs"] = y[mask]; d["pred_epochs"] = y_lce[mask]
            d["Method"] = "LCE (exp; prefix, CV-learned hypers)"; d["Regime"] = regime
            all_pred_rows.append(d)

def _scaling_feature_frame(df_in: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    if (COL_LOGP is not None) and (COL_LOGP in df_in.columns):
        feats["logP"] = df_in[COL_LOGP].astype(float)
    if (COL_LOGN is not None) and (COL_LOGN in df_in.columns):
        feats["logN"] = df_in[COL_LOGN].astype(float)
    if not feats:
        lmap = _lower_map(list(df_in.columns))
        c_params = _find_col(lmap, ["params", "param_count"])
        c_n = _find_col(lmap, ["n", "dataset_size"])
        if c_params is not None:
            feats["logP"] = np.log10(np.clip(df_in[c_params].astype(float), 1.0, None))
        if c_n is not None:
            feats["logN"] = np.log10(np.clip(df_in[c_n].astype(float), 1.0, None))
    return pd.DataFrame(feats, index=df_in.index)

def _scaling_pipe(feat_cols: List[str]) -> Pipeline:
    return Pipeline([
        ("sel", FunctionTransformer(lambda X: X[feat_cols], validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])

def _run_scaling_oof(df_in: pd.DataFrame, y_vec: np.ndarray, regime: str) -> np.ndarray:
    F = _scaling_feature_frame(df_in)
    use_cols = [c for c in ["logP", "logN"] if c in F.columns]
    if len(use_cols) == 0:
        return np.full_like(y_vec, np.nan, dtype=float)
    pipe = _scaling_pipe(use_cols)
    if regime == "CV (5-fold)":
        return run_cv(pipe, F, y_vec)
    elif regime == "LODO":
        return leave_one_group_out_oof(pipe, F, y_vec, df_in[COL_DATASET])
    elif regime == "LOMO":
        return leave_one_group_out_oof(pipe, F, y_vec, df_in[COL_MODEL])
    else:
        raise ValueError(f"Unknown regime: {regime}")

if RUN_SCALING_LAW:
    for regime in ["CV (5-fold)", "LODO", "LOMO"]:
        y_sl = _run_scaling_oof(df, y, regime)
        m = np.isfinite(y) & np.isfinite(y_sl)
        if m.any():
            all_overall_rows.append({"Method": "Scaling-Law (zero-prefix)", "Regime": regime,
                                     **base_metrics(y[m], y_sl[m])})
            d = df.loc[m].copy()
            d["true_epochs"] = y[m]; d["pred_epochs"] = y_sl[m]
            d["Method"] = "Scaling-Law (zero-prefix)"; d["Regime"] = regime
            all_pred_rows.append(d)

overall_df = pd.DataFrame(all_overall_rows)

cape_mask = overall_df["Method"] == "CAPE"
cape_print = (overall_df[cape_mask]
              [["Method","Regime","MAE","RMSE","PearsonR"]]
              .sort_values(["Regime"]))
print_table(cape_print, "=== CAPE Results ===")

base_mask = overall_df["Method"].isin([
    "CAPE (probe-only)",
    "LCE (exp; prefix, CV-learned hypers)",
    "Scaling-Law (zero-prefix)",
])
baselines_print = (overall_df[base_mask]
                   [["Method","Regime","MAE","RMSE","PearsonR"]]
                   .sort_values(["Method","Regime"]))
print_table(baselines_print, "=== Baselines ===")

pred_all = pd.concat(all_pred_rows, axis=0, ignore_index=True)

if COL_MODEL in pred_all.columns:
    pred_all = pred_all[~pred_all[COL_MODEL].apply(_hide_model)].reset_index(drop=True)

def _per_model_metrics_table(pva_df: pd.DataFrame, method: str, regime: str) -> pd.DataFrame:
    sub = pva_df[(pva_df["Method"] == method) & (pva_df["Regime"] == regime)].copy()
    if sub.empty or COL_MODEL not in sub.columns:
        return pd.DataFrame(columns=["Method","Regime","Model","N","MAE","RMSE","PearsonR"])
    rows = []
    for model_name, g in sub.groupby(COL_MODEL):
        yt = g["true_epochs"].astype(float).to_numpy()
        yp = g["pred_epochs"].astype(float).to_numpy()
        m = np.isfinite(yt) & np.isfinite(yp)
        if not m.any():
            continue
        rows.append({
            "Method": method,
            "Regime": regime,
            "Model": str(model_name),
            "N": int(m.sum()),
            "MAE": mean_absolute_error(yt[m], yp[m]),
            "RMSE": _rmse(yt[m], yp[m]),
            "PearsonR": _pearsonr_safe(yt[m], yp[m]),
        })
    return pd.DataFrame(rows)

def print_per_model_all(pred_df: pd.DataFrame):
    methods = list(pred_df["Method"].unique())
    regimes = ["CV (5-fold)", "LODO", "LOMO"]
    out_tables = []
    for reg in regimes:
        for meth in methods:
            t = _per_model_metrics_table(pred_df, meth, reg)
            if not t.empty:
                t = t.sort_values(["Method","Model"]).reset_index(drop=True)
                out_tables.append(t)
                print_table(t, f"=== Per-Model Metrics — {meth} — {reg} ===")
    return pd.concat(out_tables, axis=0, ignore_index=True) if out_tables else pd.DataFrame(
        columns=["Method","Regime","Model","N","MAE","RMSE","PearsonR"])

per_model_df = print_per_model_all(pred_all)

def _print_cape_pva_for_regime(regime_name: str):
    pva = pred_all[(pred_all["Regime"] == regime_name) & (pred_all["Method"] == "CAPE")].copy()
    if pva.empty:
        print(f"\n[INFO] No CAPE predictions found for regime '{regime_name}'.")
        return

    if (COL_OPT is not None and COL_OPT in pva.columns):
        group_keys_opt = ["Method", "Regime", COL_MODEL, COL_DATASET, COL_OPT]
        pva_by_opt = (pva.groupby(group_keys_opt, as_index=False)[["true_epochs","pred_epochs"]]
                      .mean()
                      .rename(columns={"true_epochs":"true_avg","pred_epochs":"pred_avg"})
                      .sort_values(group_keys_opt).reset_index(drop=True))
        print_table(
            pva_by_opt,
            f"=== CAPE Predicted vs Actual (Avg across LRs) — Grouped by Optimizer — {regime_name} ===\n"
            f"Columns: Method, Regime, Model, Dataset, Optimizer, true_avg, pred_avg"
        )
    else:
        print(f"\n[INFO] Optimizer column not found; skipping optimizer-averaged table for {regime_name}.")

    if (COL_LR is not None and COL_LR in pva.columns):
        group_keys_lr = ["Method", "Regime", COL_MODEL, COL_DATASET, COL_LR]
        pva_by_lr = (pva.groupby(group_keys_lr, as_index=False)[["true_epochs","pred_epochs"]]
                     .mean()
                     .rename(columns={"true_epochs":"true_avg","pred_epochs":"pred_avg"})
                     .sort_values(group_keys_lr).reset_index(drop=True))
        print_table(
            pva_by_lr,
            f"=== CAPE Predicted vs Actual (Avg across Optimizers) — Grouped by LR — {regime_name} ===\n"
            f"Columns: Method, Regime, Model, Dataset, LR, true_avg, pred_avg"
        )
    else:
        print(f"\n[INFO] Learning rate column not found; skipping LR-averaged table for {regime_name}.")

for _reg in ["CV (5-fold)", "LODO", "LOMO"]:
    _print_cape_pva_for_regime(_reg)

overall_out = "cape_vs_baselines_overall_eval.csv"
perrow_out  = "cape_vs_baselines_predictions_all.csv"
permodel_out = "cape_vs_baselines_per_model_metrics.csv"

pd.DataFrame(all_overall_rows).to_csv(overall_out, index=False)
pd.concat(all_pred_rows, axis=0, ignore_index=True).to_csv(perrow_out, index=False)
per_model_df.to_csv(permodel_out, index=False)

print(f"\n[FILES] Saved:\n - {overall_out}\n - {perrow_out}\n - {permodel_out}")

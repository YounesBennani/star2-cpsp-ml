"""
Nested CV — STAR-2, ALL MODELS BALANCED, Baseline + Diary (T1 + T2).

Fixes vs Results2/nested_cv_star2_regularized.py:
  - LogReg  : class_weight="balanced" added to grid
  - XGBoost : scale_pos_weight=2.06 (97 neg / 47 pos) added to grid
  - RF      : class_weight="balanced" (unchanged, was already there)

Output → all_balanced/Results/

Run from  Data/Star 2/all_balanced/ :
    python nested_cv_balanced.py
"""

import os, re, datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Config ──
T123_PATH    = "../STAR_T1_T3_T4_measures_QST_numeric.csv"
T2_PATH      = "../T2_deid_diary_data_numeric.csv"
ID_COL       = "studyid"
LABEL_COL    = "cpsp_t4_bin"
MAX_DAY      = 30
RANDOM_STATE = 42
OUTER_SPLITS = 5
N_REPEATS    = 10
INNER_SPLITS = 5
DEV_SIZE     = 0.20
COST_FP      = 1.0
COST_FN      = 1.0
GRID_TAU_POS = np.linspace(0.55, 0.95, 41)
GRID_TAU_NEG = np.linspace(0.05, 0.45, 41)
OUTPUT_DIR   = "Results"
FIGURES_DIR  = os.path.join(OUTPUT_DIR, "figures")
N_INTERP     = 100
SCALE_POS_WEIGHT = round(97 / 47, 4)   # 2.0638 — neg/pos ratio

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Day parsing ──
_DAY_RE = re.compile(r'_d(\d{1,2})(?=$|[^0-9])')
def _column_day(col):
    m = _DAY_RE.search(col); return int(m.group(1)) if m else None
def _base_no_day(col):
    return _DAY_RE.sub('', col).strip('_')
def _safe_slope(days_arr, vals):
    mask = ~np.isnan(vals)
    if mask.sum() < 2: return np.nan
    try: return float(np.polyfit(days_arr[mask], vals[mask], 1)[0])
    except: return np.nan

def make_t2_aggregates(df_t2, day):
    diary_cols = [c for c in df_t2.columns
                  if (dn := _column_day(c)) is not None and 1 <= dn <= day]
    if not diary_cols:
        return pd.DataFrame({ID_COL: df_t2[ID_COL].values})
    groups = {}
    for c in diary_cols:
        groups.setdefault(_base_no_day(c), []).append((_column_day(c), c))
    for b in groups: groups[b].sort(key=lambda x: x[0])
    out = pd.DataFrame({ID_COL: df_t2[ID_COL].values})
    for base, d_cols in groups.items():
        if base in {'bpi_pain_mean', 'bpi_pain_intf', 'bpi_pain_night'}: continue
        days_arr = np.array([d for d, _ in d_cols], dtype=float)
        vals = df_t2[[c for _, c in d_cols]].to_numpy(dtype=float)
        out[f"{base}__last"]  = vals[:, -1]
        out[f"{base}__mean"]  = np.nanmean(vals, axis=1)
        out[f"{base}__std"]   = np.nanstd(vals,  axis=1)
        out[f"{base}__min"]   = np.nanmin(vals,  axis=1)
        out[f"{base}__max"]   = np.nanmax(vals,  axis=1)
        out[f"{base}__slope"] = np.apply_along_axis(lambda r: _safe_slope(days_arr, r), 1, vals)
    return out

def select_t1_cols(df_t123):
    WHITELIST = {
        'site', 'survey_lang', 'gendert1', 'child_aget1',
        'rcads_anxiety_total_tt1', 'rcads_mdd_tt1',
        'ql_st1', 'pef_st1', 'prghtt1', 'prpqtt1',
        'pcq_efat1', 'pcq_pfat1', 'pdc4t1', 'pcq_appt1',
    }
    return [c for c in df_t123.columns if c in WHITELIST]

def build_X(df_t123, t2_agg_cache, ids, t1_cols, day, ref_cols=None):
    ids_str = [str(i) for i in ids]; ids_set = set(ids_str)
    t1_part = df_t123[df_t123[ID_COL].isin(ids_set)][[ID_COL] + t1_cols].copy()
    t2_sub  = t2_agg_cache[day][t2_agg_cache[day][ID_COL].isin(ids_set)].copy()
    merged  = t1_part.merge(t2_sub, on=ID_COL, how='left')
    merged  = merged.set_index(ID_COL).select_dtypes(include=np.number).reindex(ids_str)
    if ref_cols is not None:
        merged = merged.reindex(columns=ref_cols, fill_value=np.nan)
    return merged

# ── Early-decision ──
def _earliest_decision(probs, days, tau_pos, tau_neg):
    for p, d in zip(probs, days):
        if p >= tau_pos: return d, 1, float(p), False
        if p <= tau_neg: return d, 0, float(p), False
    p_last = float(probs[-1])
    return days[-1], int(p_last >= 0.5), p_last, True

def make_early_decisions(ids, y, p_all_days, tau_pos, tau_neg):
    days = sorted(p_all_days.keys()); rows = []
    for i, (pid, y_true) in enumerate(zip(ids, y)):
        probs = np.array([p_all_days[d][i] for d in days])
        dec_day, y_hat, p_at, fallback = _earliest_decision(probs, days, tau_pos, tau_neg)
        rows.append({ID_COL: str(pid), "y_true": int(y_true), "predicted": int(y_hat),
                     "decision_day": int(dec_day), "proba": float(p_at),
                     "is_fallback": bool(fallback), "correct": int(y_hat == y_true)})
    return pd.DataFrame(rows)

def _score_decisions(df):
    fp = int(((df["predicted"]==1)&(df["y_true"]==0)).sum())
    fn = int(((df["predicted"]==0)&(df["y_true"]==1)).sum())
    return COST_FP*fp + COST_FN*fn

def optimise_thresholds(ids_dev, y_dev, p_dev_all_days):
    best = (0.7, 0.3, float("inf"))
    for tp in GRID_TAU_POS:
        for tn in GRID_TAU_NEG:
            if tp <= tn: continue
            cost = _score_decisions(make_early_decisions(ids_dev, y_dev, p_dev_all_days, tp, tn))
            if cost < best[2]: best = (float(tp), float(tn), float(cost))
    return best

def compute_early_metrics(df_dec, y_train, p_train_max, stage=""):
    y_true = df_dec["y_true"].astype(int).values
    y_pred = df_dec["predicted"].astype(int).values
    y_proba = df_dec["proba"].values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    recall  = tp/(tp+fn) if (tp+fn)>0 else np.nan
    spec    = tn/(tn+fp) if (tn+fp)>0 else np.nan
    ppv     = tp/(tp+fp) if (tp+fp)>0 else np.nan
    npv     = tn/(tn+fn) if (tn+fn)>0 else np.nan
    acc     = (tp+tn)/(tp+tn+fp+fn)
    bal_acc = (recall+spec)/2 if not (np.isnan(recall) or np.isnan(spec)) else np.nan
    f1      = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else np.nan
    try: test_auroc = float(roc_auc_score(y_true, y_proba))
    except: test_auroc = np.nan
    try: test_auprc = float(average_precision_score(y_true, y_proba))
    except: test_auprc = np.nan
    try: test_brier = float(brier_score_loss(y_true, y_proba))
    except: test_brier = np.nan
    try: train_auroc = float(roc_auc_score(y_train, p_train_max))
    except: train_auroc = np.nan
    return {"stage": stage, "n_patients": int(len(df_dec)),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "Train AUROC": train_auroc, "Test AUROC": test_auroc,
            "Test AUPRC": test_auprc, "Test Brier": test_brier,
            "Accuracy": float(acc), "Balanced Accuracy": float(bal_acc),
            "Recall": float(recall), "Specificity": float(spec),
            "PPV": float(ppv), "NPV": float(npv), "F1": float(f1),
            "Mean_decision_day": float(df_dec["decision_day"].mean()),
            "Pct_decided_by_day7":  float((df_dec["decision_day"]<=7).mean()),
            "Pct_decided_by_day14": float((df_dec["decision_day"]<=14).mean()),
            "Pct_decided_by_day21": float((df_dec["decision_day"]<=21).mean()),
            "Pct_fallback_day30":   float(df_dec["is_fallback"].mean())}

def nested_cv(model_name, estimator, param_grid, df_t123, t2_agg_cache,
              ids_all, y_all, t1_cols, outer_cv, inner_cv):
    early_rows, patient_rows, thr_rows = [], [], []
    total = OUTER_SPLITS * N_REPEATS
    t_start = datetime.datetime.now()
    print(f"\n>>> Nested CV — {model_name}  ({total} outer iters)")
    days_all = list(range(1, MAX_DAY+1))
    for idx, (tr_idx, te_idx) in enumerate(outer_cv.split(ids_all, y_all)):
        rep = idx // OUTER_SPLITS + 1; fold = idx % OUTER_SPLITS + 1
        if idx == 0 or (idx+1) % 10 == 0:
            print(f"  [{model_name}] {idx+1}/{total} | rep={rep} fold={fold} | "
                  f"{str(datetime.datetime.now()-t_start).split('.')[0]}")
        train_ids_all = ids_all[tr_idx]; test_ids = ids_all[te_idx]
        y_train_all = y_all[tr_idx]; y_test = y_all[te_idx]
        inner_train_ids, dev_ids, y_inner_train, y_dev = train_test_split(
            train_ids_all, y_train_all, test_size=DEV_SIZE,
            stratify=y_train_all, random_state=RANDOM_STATE+idx)
        X_inner_max = build_X(df_t123, t2_agg_cache, inner_train_ids, t1_cols, MAX_DAY)
        ref_cols = list(X_inner_max.columns)
        search = GridSearchCV(estimator=estimator, param_grid=param_grid,
                              cv=inner_cv, scoring="roc_auc", refit=True, n_jobs=8)
        search.fit(X_inner_max, y_inner_train)
        best_pipe = search.best_estimator_
        p_train_max = best_pipe.predict_proba(X_inner_max)[:, 1]
        p_dev_days = {}; p_test_days = {}
        for d in days_all:
            X_dev_d  = build_X(df_t123, t2_agg_cache, dev_ids,  t1_cols, d, ref_cols)
            X_test_d = build_X(df_t123, t2_agg_cache, test_ids, t1_cols, d, ref_cols)
            p_dev_days[d]  = best_pipe.predict_proba(X_dev_d)[:, 1]
            p_test_days[d] = best_pipe.predict_proba(X_test_d)[:, 1]
        tau_pos, tau_neg, dev_cost = optimise_thresholds(dev_ids, y_dev, p_dev_days)
        thr_rows.append({"Model": model_name, "Repeat": rep, "Fold": fold,
                         "tau_pos": tau_pos, "tau_neg": tau_neg, "dev_cost": dev_cost})
        df_dec = make_early_decisions(test_ids, y_test, p_test_days, tau_pos, tau_neg)
        df_dec["Model"] = model_name; df_dec["Repeat"] = rep; df_dec["Fold"] = fold
        patient_rows.append(df_dec)
        metrics = compute_early_metrics(df_dec, y_inner_train, p_train_max,
                                        stage=f"fold{fold}_rep{rep}")
        metrics.update({"Model": model_name, "Repeat": rep, "Fold": fold,
                        "tau_pos": tau_pos, "tau_neg": tau_neg})
        early_rows.append(metrics)
    print(f">>> Completed {model_name} in {str(datetime.datetime.now()-t_start).split('.')[0]}")
    return (pd.DataFrame(early_rows),
            pd.concat(patient_rows, ignore_index=True),
            pd.DataFrame(thr_rows))

# ── Model definitions — ALL BALANCED ──
preprocess_lr = ColumnTransformer(
    transformers=[("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("sc",  StandardScaler())]),
                   make_column_selector(dtype_include=np.number))], remainder="drop")
preprocess_tree = ColumnTransformer(
    transformers=[("num", SimpleImputer(strategy="median"),
                   make_column_selector(dtype_include=np.number))], remainder="drop")

param_grid_lr = {
    "model__penalty":      ["elasticnet"],
    "model__solver":       ["saga"],
    "model__l1_ratio":     [0.1, 0.5, 0.9, 1.0],
    "model__C":            list(np.logspace(-4, -1, 5)),
    "model__class_weight": ["balanced"],          # ← FIXED
}
param_grid_rf = {
    "rfe__n_features_to_select": [10, 15, 20],
    "model__n_estimators":       [300],
    "model__max_depth":          [2, 3, 4],
    "model__min_samples_leaf":   [10, 15, 20],
    "model__max_features":       ["sqrt", 0.3],
    "model__class_weight":       ["balanced"],    # unchanged
}
param_grid_xgb = {
    "rfe__n_features_to_select": [10, 15, 20],
    "model__n_estimators":       [200],
    "model__learning_rate":      [0.03, 0.1],
    "model__max_depth":          [2, 3],
    "model__subsample":          [0.8],
    "model__colsample_bytree":   [0.8],
    "model__min_child_weight":   [5, 10],
    "model__reg_alpha":          [0.1, 0.5],
    "model__reg_lambda":         [1, 5, 10],
    "model__scale_pos_weight":   [SCALE_POS_WEIGHT],  # ← FIXED
}

rfe_rf  = RFE(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), step=0.3)
rfe_xgb = RFE(XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=RANDOM_STATE), step=0.3)

models_to_run = [
    {"name": "LogReg_ElasticNet",
     "pipeline": Pipeline([("preprocess", preprocess_lr),
                           ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=10000))]),
     "grid": param_grid_lr},
    {"name": "RandomForest_RFE",
     "pipeline": Pipeline([("preprocess", preprocess_tree), ("rfe", rfe_rf),
                           ("model", RandomForestClassifier(random_state=RANDOM_STATE))]),
     "grid": param_grid_rf},
    {"name": "XGBoost_RFE",
     "pipeline": Pipeline([("preprocess", preprocess_tree), ("rfe", rfe_xgb),
                           ("model", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE))]),
     "grid": param_grid_xgb},
]

if __name__ == "__main__":
    print(f"scale_pos_weight for XGB: {SCALE_POS_WEIGHT}")
    print("Loading data...")
    t123 = pd.read_csv(T123_PATH); t2 = pd.read_csv(T2_PATH)
    t123.columns = t123.columns.str.lower(); t2.columns = t2.columns.str.lower()
    valid = t123[["pdc4t4","ql_st4"]].notna().all(axis=1)
    cond  = (t123["pdc4t4"] >= 3) & (t123["ql_st4"] < 74.9)
    t123[LABEL_COL] = np.where(valid, cond.astype(float), np.nan)
    t123 = t123.loc[t123[LABEL_COL].notna()].copy()
    t123[ID_COL] = t123[ID_COL].astype(str); t2[ID_COL] = t2[ID_COL].astype(str)
    t123 = t123[t123[ID_COL].isin(set(t2[ID_COL]))].copy()
    t2   = t2[t2[ID_COL].isin(set(t123[ID_COL]))].copy()
    ids_all = t123[ID_COL].values; y_all = t123[LABEL_COL].astype(int).values
    t1_cols = select_t1_cols(t123); prevalence = float(y_all.mean())
    print(f"N={len(ids_all)}, CPSP={y_all.sum()} ({prevalence:.1%}), T1 feats={len(t1_cols)}")

    print(f"\nPrecomputing T2 aggregates 1..{MAX_DAY}...")
    t2_agg_cache = {}
    for d in range(1, MAX_DAY+1):
        t2_agg_cache[d] = make_t2_aggregates(t2, d)
        if d % 5 == 0: print(f"  day {d}/{MAX_DAY}")

    outer_cv = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    inner_cv = RepeatedStratifiedKFold(n_splits=INNER_SPLITS, n_repeats=1, random_state=RANDOM_STATE)

    all_early, all_patients, all_thr = [], [], []
    for m in models_to_run:
        df_e, df_p, df_t = nested_cv(m["name"], m["pipeline"], m["grid"],
                                      t123, t2_agg_cache, ids_all, y_all, t1_cols,
                                      outer_cv, inner_cv)
        all_early.append(df_e); all_patients.append(df_p); all_thr.append(df_t)

    early_all    = pd.concat(all_early,    ignore_index=True)
    patients_all = pd.concat(all_patients, ignore_index=True)
    thr_all      = pd.concat(all_thr,      ignore_index=True)

    metric_cols = ["Train AUROC","Test AUROC","Test AUPRC","Test Brier","Accuracy",
                   "Balanced Accuracy","Recall","Specificity","PPV","NPV","F1",
                   "Mean_decision_day","Pct_decided_by_day7","Pct_decided_by_day14",
                   "Pct_decided_by_day21","Pct_fallback_day30"]
    summary = early_all.groupby("Model")[metric_cols].agg(["mean","std"]).reset_index()
    summary.columns = ["Model"] + [f"{c[0]} {c[1]}" for c in summary.columns if c[0]!="Model"]

    print("\n"+"="*60+"\nSUMMARY — ALL BALANCED, Baseline+Diary\n"+"="*60)
    print(summary[["Model","Test AUROC mean","Test AUROC std","Balanced Accuracy mean",
                   "Mean_decision_day mean","Pct_fallback_day30 mean"]].to_string(index=False))

    out_file = os.path.join(OUTPUT_DIR, "results.xlsx")
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        summary.to_excel(writer,      sheet_name="Summary",           index=False)
        early_all.to_excel(writer,    sheet_name="PerFold",           index=False)
        patients_all.to_excel(writer, sheet_name="PerPatient",        index=False)
        thr_all.to_excel(writer,      sheet_name="OptimalThresholds", index=False)
    print(f"\nResults → {out_file}")

    # ── Figures ──
    colors = {"LogReg_ElasticNet":"#1f77b4","RandomForest_RFE":"#ff7f0e","XGBoost_RFE":"#2ca02c"}
    model_names = [m["name"] for m in models_to_run]
    mean_fpr = np.linspace(0,1,N_INTERP); mean_recall = np.linspace(0,1,N_INTERP)

    def pool_by_repeat(preds_df):
        return {rep: (g["y_true"].values, g["y_proba"].values)
                for rep, g in preds_df.groupby("Repeat")}

    # Decision day
    fig, ax = plt.subplots(figsize=(9,5))
    for mn in model_names:
        sub = patients_all[patients_all["Model"]==mn].copy(); c = colors.get(mn,"gray")
        by_fold = sub.groupby(["Repeat","Fold","decision_day"]).size().reset_index(name="n")
        n_per   = sub.groupby(["Repeat","Fold"]).size().reset_index(name="total")
        by_fold = by_fold.merge(n_per, on=["Repeat","Fold"])
        by_fold["pct"] = by_fold["n"]/by_fold["total"]*100
        pivot = by_fold.groupby("decision_day")["pct"].mean().sort_index().cumsum()
        ax.plot(pivot.index, pivot.values, "o-", color=c, lw=2, label=mn)
    ax.axvline(x=MAX_DAY, color="gray", ls=":", lw=1)
    ax.set_xlabel("Day post-surgery"); ax.set_ylabel("% patients decided (cumulative)")
    ax.set_title("Early-Decision Rate — All Balanced, Baseline+Diary", fontweight="bold")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR,"decision_day_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(7,6))
    for mn in model_names:
        sub  = patients_all[patients_all["Model"]==mn]
        pool = pool_by_repeat(sub.rename(columns={"proba":"y_proba"}))
        tprs, aucs = [], []
        for yt, yp in pool.values():
            fpr_, tpr_, _ = roc_curve(yt, yp)
            interp_tpr = np.interp(mean_fpr, fpr_, tpr_); interp_tpr[0] = 0.0
            tprs.append(interp_tpr); aucs.append(roc_auc_score(yt, yp))
        tprs = np.array(tprs); mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        c = colors.get(mn,"gray")
        ax_roc.plot(mean_fpr, mean_tpr, color=c, lw=2,
                    label=f"{mn} (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
        ax_roc.fill_between(mean_fpr, np.percentile(tprs,2.5,axis=0),
                            np.percentile(tprs,97.5,axis=0), color=c, alpha=0.1)
    ax_roc.plot([0,1],[0,1],"k--",lw=1)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC — All Balanced, Baseline+Diary", fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9); ax_roc.set_aspect("equal")
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(FIGURES_DIR,"ROC_combined.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_roc)

    # PR
    fig_pr, ax_pr = plt.subplots(figsize=(7,6))
    for mn in model_names:
        sub  = patients_all[patients_all["Model"]==mn]
        pool = pool_by_repeat(sub.rename(columns={"proba":"y_proba"}))
        precs, aps = [], []
        for yt, yp in pool.values():
            prec_, rec_, _ = precision_recall_curve(yt, yp)
            precs.append(np.interp(mean_recall, rec_[::-1], prec_[::-1]))
            aps.append(average_precision_score(yt, yp))
        precs = np.array(precs); c = colors.get(mn,"gray")
        ax_pr.plot(mean_recall, np.mean(precs,axis=0), color=c, lw=2,
                   label=f"{mn} (AP={np.mean(aps):.3f}±{np.std(aps):.3f})")
        ax_pr.fill_between(mean_recall, np.percentile(precs,2.5,axis=0),
                           np.percentile(precs,97.5,axis=0), color=c, alpha=0.1)
    ax_pr.axhline(prevalence, color="k", ls="--", lw=1, label=f"Prevalence={prevalence:.2f}")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR — All Balanced, Baseline+Diary", fontweight="bold")
    ax_pr.legend(loc="upper right", fontsize=9); fig_pr.tight_layout()
    fig_pr.savefig(os.path.join(FIGURES_DIR,"PR_combined.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_pr)

    # Calibration
    N_CAL = 10; cal_centers = np.linspace(1/(2*N_CAL), 1-1/(2*N_CAL), N_CAL)
    fig_cal, ax_cal = plt.subplots(figsize=(7,6))
    for mn in model_names:
        sub  = patients_all[patients_all["Model"]==mn]
        pool = pool_by_repeat(sub.rename(columns={"proba":"y_proba"}))
        fracs = []
        for yt, yp in pool.values():
            try:
                frac_, mpv_ = calibration_curve(yt, yp, n_bins=N_CAL, strategy="uniform")
                if len(mpv_) >= 2:
                    fi = interp1d(mpv_, frac_, kind="linear", bounds_error=False,
                                  fill_value=(frac_[0], frac_[-1]))
                    fracs.append(fi(cal_centers))
            except: continue
        if fracs:
            fracs = np.array(fracs); c = colors.get(mn,"gray")
            ax_cal.plot(cal_centers, np.nanmean(fracs,axis=0), "s-", color=c, lw=2, label=mn)
            ax_cal.fill_between(cal_centers, np.nanpercentile(fracs,2.5,axis=0),
                                np.nanpercentile(fracs,97.5,axis=0), color=c, alpha=0.1)
    ax_cal.plot([0,1],[0,1],"k--",lw=1,label="Perfect")
    ax_cal.set_xlabel("Mean Predicted Probability"); ax_cal.set_ylabel("Observed Frequency")
    ax_cal.set_title("Calibration — All Balanced, Baseline+Diary", fontweight="bold")
    ax_cal.legend(loc="lower right", fontsize=9); ax_cal.set_aspect("equal")
    fig_cal.tight_layout()
    fig_cal.savefig(os.path.join(FIGURES_DIR,"Calibration_combined.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_cal)

    print(f"\nAll figures → {FIGURES_DIR}\nDone!")

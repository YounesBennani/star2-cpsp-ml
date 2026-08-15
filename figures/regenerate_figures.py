"""Regenerate figures from results.xlsx with corrected CIs and decision day plot."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from scipy.interpolate import interp1d

RESULTS_PATH = "Results/results.xlsx"
FIGURES_DIR  = "Results/figures"
N_INTERP     = 100
N_CAL        = 10
os.makedirs(FIGURES_DIR, exist_ok=True)

patients_all = pd.read_excel(RESULTS_PATH, sheet_name="PerPatient")
model_names  = sorted(patients_all["Model"].unique())
prevalence   = patients_all["y_true"].mean()

colors = {"LogReg_ElasticNet":"#1f77b4", "RandomForest_RFE":"#ff7f0e", "XGBoost_RFE":"#2ca02c"}
mean_fpr    = np.linspace(0, 1, N_INTERP)
mean_recall = np.linspace(0, 1, N_INTERP)
cal_centers = np.linspace(1/(2*N_CAL), 1 - 1/(2*N_CAL), N_CAL)

def fold_curves_roc(sub):
    tprs, aucs = [], []
    for (rep, fold), g in sub.groupby(["Repeat","Fold"]):
        yt, yp = g["y_true"].values, g["proba"].values
        if len(np.unique(yt)) < 2: continue
        fpr_, tpr_, _ = roc_curve(yt, yp)
        t = np.interp(mean_fpr, fpr_, tpr_); t[0] = 0.0
        tprs.append(t); aucs.append(roc_auc_score(yt, yp))
    return np.array(tprs), np.array(aucs)

def fold_curves_pr(sub):
    precs, aps = [], []
    for (rep, fold), g in sub.groupby(["Repeat","Fold"]):
        yt, yp = g["y_true"].values, g["proba"].values
        if len(np.unique(yt)) < 2: continue
        prec_, rec_, _ = precision_recall_curve(yt, yp)
        precs.append(np.interp(mean_recall, rec_[::-1], prec_[::-1]))
        aps.append(average_precision_score(yt, yp))
    return np.array(precs), np.array(aps)

def ci_band(curves):
    n = len(curves)
    mu = np.mean(curves, axis=0)
    se = np.std(curves, axis=0) / np.sqrt(n)
    return mu, mu - 1.96*se, mu + 1.96*se

# ── Decision day distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for mn in model_names:
    sub = patients_all[patients_all["Model"] == mn].copy()
    c   = colors.get(mn, "gray")
    total = len(sub)
    day_counts = sub.groupby("decision_day").size().sort_index()
    cum_pct    = day_counts.cumsum() / total * 100
    ax.plot(cum_pct.index, cum_pct.values, "o-", color=c, lw=2, label=mn)
ax.set_xlim(1, 30); ax.set_ylim(0, 100)
ax.set_xlabel("Day post-surgery", fontsize=12)
ax.set_ylabel("Cumulative % of patients decided", fontsize=12)
ax.set_title("Early-Decision Rate — Baseline+Diary", fontweight="bold")
ax.set_xticks([1,5,10,15,20,25,30]); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "decision_day_distribution.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("✓ decision_day_distribution")

# ── ROC ──────────────────────────────────────────────────────────────────────
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
for mn in model_names:
    sub = patients_all[patients_all["Model"] == mn]
    c   = colors.get(mn, "gray")
    tprs, aucs = fold_curves_roc(sub)
    if len(tprs) == 0: continue
    n = len(aucs)
    mu_tpr, lo_tpr, hi_tpr = ci_band(tprs)
    mu_tpr[-1] = 1.0
    mean_auc = np.mean(aucs); se_auc = np.std(aucs)/np.sqrt(n)
    ax_roc.plot(mean_fpr, mu_tpr, color=c, lw=2,
                label=f"{mn}\nAUC={mean_auc:.3f} [±{1.96*se_auc:.3f}]")
    ax_roc.fill_between(mean_fpr, lo_tpr, hi_tpr, color=c, alpha=0.15)
ax_roc.plot([0,1],[0,1],"k--",lw=1)
ax_roc.set_xlabel("False Positive Rate", fontsize=12)
ax_roc.set_ylabel("True Positive Rate", fontsize=12)
ax_roc.set_title("ROC — Baseline+Diary", fontweight="bold")
ax_roc.legend(loc="lower right", fontsize=8)
ax_roc.set_xlim(0,1); ax_roc.set_ylim(0,1)
fig_roc.tight_layout()
fig_roc.savefig(os.path.join(FIGURES_DIR, "ROC_combined.png"), dpi=300, bbox_inches="tight")
plt.close(fig_roc)
print("✓ ROC_combined")

# ── PR ───────────────────────────────────────────────────────────────────────
fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
for mn in model_names:
    sub = patients_all[patients_all["Model"] == mn]
    c   = colors.get(mn, "gray")
    precs, aps = fold_curves_pr(sub)
    if len(precs) == 0: continue
    n = len(aps)
    mu_prec, lo_prec, hi_prec = ci_band(precs)
    mean_ap = np.mean(aps); se_ap = np.std(aps)/np.sqrt(n)
    ax_pr.plot(mean_recall, mu_prec, color=c, lw=2,
               label=f"{mn}\nAP={mean_ap:.3f} [±{1.96*se_ap:.3f}]")
    ax_pr.fill_between(mean_recall, lo_prec, hi_prec, color=c, alpha=0.15)
ax_pr.axhline(prevalence, color="k", ls="--", lw=1, label=f"Prevalence={prevalence:.2f}")
ax_pr.set_xlabel("Recall", fontsize=12); ax_pr.set_ylabel("Precision", fontsize=12)
ax_pr.set_title("Precision-Recall — Baseline+Diary", fontweight="bold")
ax_pr.legend(loc="upper right", fontsize=8)
ax_pr.set_xlim(0,1); ax_pr.set_ylim(0,1)
fig_pr.tight_layout()
fig_pr.savefig(os.path.join(FIGURES_DIR, "PR_combined.png"), dpi=300, bbox_inches="tight")
plt.close(fig_pr)
print("✓ PR_combined")

# ── Calibration (pooled all folds) ──────────────────────────────────────────
fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
for mn in model_names:
    sub = patients_all[patients_all["Model"] == mn]
    c   = colors.get(mn, "gray")
    fracs = []
    for (rep, fold), g in sub.groupby(["Repeat","Fold"]):
        yt, yp = g["y_true"].values, g["proba"].values
        if len(np.unique(yt)) < 2: continue
        try:
            frac_, mpv_ = calibration_curve(yt, yp, n_bins=N_CAL, strategy="uniform")
            if len(mpv_) >= 2:
                fi = interp1d(mpv_, frac_, kind="linear", bounds_error=False,
                              fill_value=(frac_[0], frac_[-1]))
                fracs.append(fi(cal_centers))
        except: continue
    if not fracs: continue
    fracs = np.array(fracs)
    mu_cal, lo_cal, hi_cal = ci_band(fracs)
    ax_cal.plot(cal_centers, mu_cal, "s-", color=c, lw=2, label=mn)
    ax_cal.fill_between(cal_centers, lo_cal, hi_cal, color=c, alpha=0.15)
ax_cal.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
ax_cal.set_xlabel("Mean Predicted Probability", fontsize=12)
ax_cal.set_ylabel("Observed Frequency", fontsize=12)
ax_cal.set_title("Calibration — Baseline+Diary", fontweight="bold")
ax_cal.legend(loc="lower right", fontsize=9)
ax_cal.set_xlim(0,1); ax_cal.set_ylim(0,1)
fig_cal.tight_layout()
fig_cal.savefig(os.path.join(FIGURES_DIR, "Calibration_combined.png"), dpi=300, bbox_inches="tight")
plt.close(fig_cal)
print("✓ Calibration_combined")

print(f"\nAll figures → {FIGURES_DIR}/")

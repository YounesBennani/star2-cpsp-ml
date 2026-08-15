"""Build the combined Figure 2 (a) ROC, (b) PR, (c) Calibration — single multi-panel
figure in npj Digital Medicine style. Reads Results/results.xlsx (PerPatient)."""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from scipy.interpolate import interp1d

RESULTS_PATH = "Results/results.xlsx"
FIGURES_DIR  = "Results/figures"
OUT          = os.path.join(FIGURES_DIR, "Figure2_combined.png")
N_INTERP, N_CAL = 100, 10
os.makedirs(FIGURES_DIR, exist_ok=True)

patients = pd.read_excel(RESULTS_PATH, sheet_name="PerPatient")
model_names = sorted(patients["Model"].unique())
prevalence  = patients["y_true"].mean()

COLORS = {"LogReg_ElasticNet": "#1f77b4", "RandomForest_RFE": "#ff7f0e", "XGBoost_RFE": "#2ca02c"}
PRETTY = {"LogReg_ElasticNet": "Logistic regression", "RandomForest_RFE": "Random Forest", "XGBoost_RFE": "XGBoost"}
ORDER  = ["RandomForest_RFE", "XGBoost_RFE", "LogReg_ElasticNet"]
models = [m for m in ORDER if m in model_names] + [m for m in model_names if m not in ORDER]

mean_fpr    = np.linspace(0, 1, N_INTERP)
mean_recall = np.linspace(0, 1, N_INTERP)
cal_centers = np.linspace(1/(2*N_CAL), 1 - 1/(2*N_CAL), N_CAL)

def ci_band(curves):
    n = len(curves); mu = np.mean(curves, axis=0); se = np.std(curves, axis=0) / np.sqrt(n)
    return mu, mu - 1.96*se, mu + 1.96*se

def fold_roc(sub):
    tprs, aucs = [], []
    for _, g in sub.groupby(["Repeat", "Fold"]):
        yt, yp = g["y_true"].values, g["proba"].values
        if len(np.unique(yt)) < 2: continue
        fpr_, tpr_, _ = roc_curve(yt, yp); t = np.interp(mean_fpr, fpr_, tpr_); t[0] = 0.0
        tprs.append(t); aucs.append(roc_auc_score(yt, yp))
    return np.array(tprs), np.array(aucs)

def fold_pr(sub):
    precs, aps = [], []
    for _, g in sub.groupby(["Repeat", "Fold"]):
        yt, yp = g["y_true"].values, g["proba"].values
        if len(np.unique(yt)) < 2: continue
        p_, r_, _ = precision_recall_curve(yt, yp)
        precs.append(np.interp(mean_recall, r_[::-1], p_[::-1])); aps.append(average_precision_score(yt, yp))
    return np.array(precs), np.array(aps)

def fold_cal(sub):
    fracs = []
    for _, g in sub.groupby(["Repeat", "Fold"]):
        yt, yp = g["y_true"].values, g["proba"].values
        if len(np.unique(yt)) < 2: continue
        try:
            frac_, mpv_ = calibration_curve(yt, yp, n_bins=N_CAL, strategy="uniform")
            if len(mpv_) >= 2:
                fi = interp1d(mpv_, frac_, kind="linear", bounds_error=False, fill_value=(frac_[0], frac_[-1]))
                fracs.append(fi(cal_centers))
        except: continue
    return np.array(fracs)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
axa, axb, axc = axes

# ── (a) ROC ──────────────────────────────────────────────────────────────────
for mn in models:
    sub = patients[patients["Model"] == mn]; c = COLORS.get(mn, "gray")
    tprs, aucs = fold_roc(sub)
    if len(tprs) == 0: continue
    mu, lo, hi = ci_band(tprs); mu[-1] = 1.0
    auc, se = np.mean(aucs), np.std(aucs)/np.sqrt(len(aucs))
    axa.plot(mean_fpr, mu, color=c, lw=2.2, label=f"{PRETTY.get(mn, mn)} (AUC {auc:.3f} [±{1.96*se:.3f}])")
    axa.fill_between(mean_fpr, lo, hi, color=c, alpha=0.15)
axa.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC 0.50)")
axa.set_xlabel("False positive rate", fontsize=12)
axa.set_ylabel("True positive rate", fontsize=12)
axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.legend(loc="lower right", fontsize=8.5, frameon=False)

# ── (b) PR ───────────────────────────────────────────────────────────────────
for mn in models:
    sub = patients[patients["Model"] == mn]; c = COLORS.get(mn, "gray")
    precs, aps = fold_pr(sub)
    if len(precs) == 0: continue
    mu, lo, hi = ci_band(precs)
    ap, se = np.mean(aps), np.std(aps)/np.sqrt(len(aps))
    axb.plot(mean_recall, mu, color=c, lw=2.2, label=f"{PRETTY.get(mn, mn)} (AP {ap:.3f} [±{1.96*se:.3f}])")
    axb.fill_between(mean_recall, lo, hi, color=c, alpha=0.15)
axb.axhline(prevalence, color="k", ls="--", lw=1, label=f"Prevalence ({prevalence:.2f})")
axb.set_xlabel("Recall", fontsize=12); axb.set_ylabel("Precision", fontsize=12)
axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.legend(loc="upper right", fontsize=8.5, frameon=False)

# ── (c) Calibration ──────────────────────────────────────────────────────────
for mn in models:
    sub = patients[patients["Model"] == mn]; c = COLORS.get(mn, "gray")
    fracs = fold_cal(sub)
    if len(fracs) == 0: continue
    mu, lo, hi = ci_band(fracs)
    axc.plot(cal_centers, mu, "s-", color=c, lw=2.2, ms=5, label=PRETTY.get(mn, mn))
    axc.fill_between(cal_centers, lo, hi, color=c, alpha=0.15)
axc.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
axc.set_xlabel("Mean predicted probability", fontsize=12)
axc.set_ylabel("Observed frequency", fontsize=12)
axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.legend(loc="upper left", fontsize=8.5, frameon=False)

# ── Panel letters (lowercase bold, npj style) ────────────────────────────────
for ax, letter in zip(axes, ["a", "b", "c"]):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top", ha="left")
    ax.grid(True, alpha=0.25)

fig.tight_layout(w_pad=2.5)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}")

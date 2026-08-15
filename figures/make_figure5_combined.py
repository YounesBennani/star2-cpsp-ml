"""Build combined Figure 5: (a) cumulative early-decision rate (3 classifiers),
(b) predicted CPSP probability at the decision day by true status (Random Forest).
Reads Results/results.xlsx (PerPatient)."""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_PATH = "Results/results.xlsx"
FIGURES_DIR  = "Results/figures"
OUT          = os.path.join(FIGURES_DIR, "Figure5_combined.png")
ID_COL = "studyid"
os.makedirs(FIGURES_DIR, exist_ok=True)

patients = pd.read_excel(RESULTS_PATH, sheet_name="PerPatient")

COLORS = {"LogReg_ElasticNet": "#1f77b4", "RandomForest_RFE": "#ff7f0e", "XGBoost_RFE": "#2ca02c"}
PRETTY = {"LogReg_ElasticNet": "Logistic regression", "RandomForest_RFE": "Random Forest", "XGBoost_RFE": "XGBoost"}
ORDER  = ["RandomForest_RFE", "XGBoost_RFE", "LogReg_ElasticNet"]
models = [m for m in ORDER if m in patients["Model"].unique()]
C_POS, C_NEG = "#d62728", "#1f77b4"

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
# 2-column page-friendly width (~9.4in → fits a portrait page when scaled to margins)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 4.6))

# ── (a) Cumulative early-decision rate ───────────────────────────────────────
for mn in models:
    sub = patients[patients["Model"] == mn]
    total = len(sub)
    day_counts = sub.groupby("decision_day").size().sort_index()
    cum_pct = day_counts.cumsum() / total * 100
    axa.plot(cum_pct.index, cum_pct.values, "o-", color=COLORS[mn], lw=2.2, ms=4, label=PRETTY[mn])
axa.set_xlim(1, 30); axa.set_ylim(0, 100)
axa.set_xticks([1, 5, 10, 15, 20, 25, 30])
axa.set_xlabel("Day post-surgery", fontsize=12)
axa.set_ylabel("Cumulative % of patients decided", fontsize=12)
axa.legend(loc="lower right", fontsize=9, frameon=False)
axa.grid(True, alpha=0.25)

# ── (b) Decision-day probability by true status (Random Forest) ──────────────
rf = patients[patients["Model"] == "RandomForest_RFE"]
pat = rf.groupby(ID_COL).agg(proba=("proba", "mean"), y_true=("y_true", "first")).reset_index()
p_pos = pat[pat["y_true"] == 1]["proba"].values
p_neg = pat[pat["y_true"] == 0]["proba"].values
U, pval = stats.mannwhitneyu(p_pos, p_neg, alternative="greater")
r = (2*U)/(len(p_pos)*len(p_neg)) - 1
pstr = "p < 0.001" if pval < 0.001 else (f"p = {pval:.3f}")

bp = axb.boxplot([p_neg, p_pos], tick_labels=["No CPSP\n(n=97)", "CPSP\n(n=47)"],
                 patch_artist=True, widths=0.6, medianprops=dict(color="black", linewidth=2),
                 flierprops=dict(marker="o", markersize=3, alpha=0.4))
bp["boxes"][0].set_facecolor(C_NEG); bp["boxes"][0].set_alpha(0.65)
bp["boxes"][1].set_facecolor(C_POS); bp["boxes"][1].set_alpha(0.65)
# jittered points
for i, vals in enumerate([p_neg, p_pos], start=1):
    x = np.random.default_rng(0).normal(i, 0.05, size=len(vals))
    axb.plot(x, vals, "o", color="gray", ms=2.5, alpha=0.35)
axb.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
ytop = max(p_pos.max(), p_neg.max())
axb.plot([1, 2], [ytop*1.05, ytop*1.05], color="black", lw=1.2)
axb.annotate(f"{pstr}  (r = {r:.2f})", xy=(1.5, ytop*1.08), ha="center", fontsize=10.5, fontweight="bold")
axb.set_ylim(0, 1.18)
axb.set_ylabel("Predicted CPSP probability at decision day", fontsize=12)
axb.grid(True, axis="y", alpha=0.25)

for ax, letter in zip((axa, axb), ["a", "b"]):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top", ha="left")

fig.tight_layout(w_pad=3)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}")
print(f"RF: Mann-Whitney U={U:.0f}, {pstr}, r={r:.3f}; median CPSP+={np.median(p_pos):.3f}, CPSP-={np.median(p_neg):.3f}")

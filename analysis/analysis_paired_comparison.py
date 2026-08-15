"""
Paired fold-by-fold comparison: Baseline+Diary vs Diary-Only (All Balanced).

For each outer fold (Repeat × Fold), both conditions used the exact same test
patients. We compare AUROC fold-by-fold using:
  - Wilcoxon signed-rank test (paired, non-parametric)
  - Paired t-test
  - Mean difference + 95% CI

This cancels out the variance from fold assignment and isolates the
effect of adding baseline (T1) features.

Run from  Data/Star 2/all_balanced/ :
    python analysis_paired_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = "Analysis/paired_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ──
print("Loading Results/results.xlsx  (Baseline+Diary)...")
df_bd = pd.read_excel("Results/results.xlsx",          sheet_name="PerFold")
print("Loading Results_diary_only/results.xlsx  (Diary Only)...")
df_do = pd.read_excel("Results_diary_only/results.xlsx", sheet_name="PerFold")

models = df_bd["Model"].unique()
print(f"Models: {list(models)}")
print(f"Folds per model: {df_bd.groupby('Model').size().iloc[0]}")

results_rows = []

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
if len(models) == 1:
    axes = [axes]

colors = {"LogReg_ElasticNet": "#1f77b4",
          "RandomForest_RFE":  "#ff7f0e",
          "XGBoost_RFE":       "#2ca02c"}

for ax, model in zip(axes, models):
    bd = df_bd[df_bd["Model"] == model][["Repeat","Fold","Test AUROC"]].rename(
        columns={"Test AUROC": "AUROC_BD"})
    do = df_do[df_do["Model"] == model][["Repeat","Fold","Test AUROC"]].rename(
        columns={"Test AUROC": "AUROC_DO"})

    merged = bd.merge(do, on=["Repeat","Fold"])
    merged["Delta"] = merged["AUROC_BD"] - merged["AUROC_DO"]   # positive = BD better

    n      = len(merged)
    mean_d = merged["Delta"].mean()
    std_d  = merged["Delta"].std()
    se     = std_d / np.sqrt(n)
    ci95   = 1.96 * se
    pct_positive = (merged["Delta"] > 0).mean() * 100

    # Wilcoxon signed-rank (non-parametric paired test)
    stat_w, p_w = stats.wilcoxon(merged["AUROC_BD"], merged["AUROC_DO"],
                                  alternative="greater")

    # Paired t-test
    stat_t, p_t = stats.ttest_rel(merged["AUROC_BD"], merged["AUROC_DO"],
                                   alternative="greater")

    pstr_w = ("p < 0.001" if p_w < 0.001 else
              "p < 0.01"  if p_w < 0.01  else
              "p < 0.05"  if p_w < 0.05  else
              f"p = {p_w:.3f}")
    pstr_t = ("p < 0.001" if p_t < 0.001 else
              "p < 0.01"  if p_t < 0.01  else
              "p < 0.05"  if p_t < 0.05  else
              f"p = {p_t:.3f}")

    results_rows.append({
        "Model":             model,
        "N_folds":           n,
        "Mean_AUROC_BD":     float(df_bd[df_bd["Model"]==model]["Test AUROC"].mean()),
        "Mean_AUROC_DO":     float(df_do[df_do["Model"]==model]["Test AUROC"].mean()),
        "Mean_Delta":        float(mean_d),
        "Std_Delta":         float(std_d),
        "95CI_lower":        float(mean_d - ci95),
        "95CI_upper":        float(mean_d + ci95),
        "Pct_folds_BD_wins": float(pct_positive),
        "Wilcoxon_stat":     float(stat_w),
        "Wilcoxon_p":        float(p_w),
        "Wilcoxon_p_str":    pstr_w,
        "PairedT_stat":      float(stat_t),
        "PairedT_p":         float(p_t),
        "PairedT_p_str":     pstr_t,
    })

    print(f"\n[{model}]")
    print(f"  BD mean AUROC : {results_rows[-1]['Mean_AUROC_BD']:.4f}")
    print(f"  DO mean AUROC : {results_rows[-1]['Mean_AUROC_DO']:.4f}")
    print(f"  Mean Δ (BD−DO): {mean_d:+.4f}  (95% CI [{mean_d-ci95:+.4f}, {mean_d+ci95:+.4f}])")
    print(f"  % folds BD > DO: {pct_positive:.1f}%")
    print(f"  Wilcoxon : {pstr_w}")
    print(f"  Paired t : {pstr_t}")

    # ── Plot: scatter BD vs DO AUROC per fold ──
    c = colors.get(model, "gray")
    ax.scatter(merged["AUROC_DO"], merged["AUROC_BD"], alpha=0.5, color=c, s=30)
    lims = [min(merged["AUROC_DO"].min(), merged["AUROC_BD"].min()) - 0.02,
            max(merged["AUROC_DO"].max(), merged["AUROC_BD"].max()) + 0.02]
    ax.plot(lims, lims, "k--", lw=1, label="Equal")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("AUROC — Diary Only"); ax.set_ylabel("AUROC — Baseline+Diary")
    ax.set_title(f"{model}\nΔ={mean_d:+.3f}  {pstr_w}", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    sig = "*" if min(p_w, p_t) < 0.05 else "ns"
    ax.text(0.05, 0.95, sig, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top",
            color="red" if sig != "ns" else "gray")

fig.suptitle("Paired fold-by-fold: Baseline+Diary vs Diary Only (All Balanced)\n"
             "Each dot = one outer fold. Points above diagonal = BD wins.",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "paired_scatter.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "paired_scatter.pdf"), bbox_inches="tight")
plt.close(fig)

# ── Delta distribution plot ──
fig2, axes2 = plt.subplots(1, len(models), figsize=(5*len(models), 4), sharey=True)
if len(models) == 1:
    axes2 = [axes2]

for ax, model, row in zip(axes2, models, results_rows):
    bd = df_bd[df_bd["Model"]==model][["Repeat","Fold","Test AUROC"]].rename(columns={"Test AUROC":"AUROC_BD"})
    do = df_do[df_do["Model"]==model][["Repeat","Fold","Test AUROC"]].rename(columns={"Test AUROC":"AUROC_DO"})
    merged = bd.merge(do, on=["Repeat","Fold"])
    merged["Delta"] = merged["AUROC_BD"] - merged["AUROC_DO"]
    c = colors.get(model, "gray")
    ax.hist(merged["Delta"], bins=20, color=c, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, ls="--", label="No difference")
    ax.axvline(row["Mean_Delta"], color="red", lw=2, label=f"Mean={row['Mean_Delta']:+.3f}")
    ax.set_xlabel("ΔAUROC (Baseline+Diary − Diary Only)")
    ax.set_ylabel("Count" if ax == axes2[0] else "")
    ax.set_title(f"{model}\n{row['Wilcoxon_p_str']}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

fig2.suptitle("Distribution of ΔAUROC per fold (All Balanced)", fontsize=12, fontweight="bold")
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "delta_distribution.png"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(OUTPUT_DIR, "delta_distribution.pdf"), bbox_inches="tight")
plt.close(fig2)

# ── Export ──
results_df = pd.DataFrame(results_rows)
out_file = os.path.join(OUTPUT_DIR, "paired_comparison.xlsx")
with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Summary", index=False)
    # Also export raw fold-level deltas
    all_deltas = []
    for model in models:
        bd = df_bd[df_bd["Model"]==model][["Repeat","Fold","Test AUROC"]].rename(columns={"Test AUROC":"AUROC_BD"})
        do = df_do[df_do["Model"]==model][["Repeat","Fold","Test AUROC"]].rename(columns={"Test AUROC":"AUROC_DO"})
        m  = bd.merge(do, on=["Repeat","Fold"])
        m["Delta"] = m["AUROC_BD"] - m["AUROC_DO"]
        m["Model"] = model
        all_deltas.append(m)
    pd.concat(all_deltas).to_excel(writer, sheet_name="PerFold_Deltas", index=False)

print(f"\nSaved → {out_file}")
print(f"Figures → {OUTPUT_DIR}/")
print("Done!")

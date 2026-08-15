"""Re-draw auroc_vs_day figures with correct 95% CI band (mean ± 1.96·SD/√n).
Reads existing PerFold data — no CV re-run needed."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"LogReg_ElasticNet": "#1f77b4", "RandomForest_RFE": "#ff7f0e", "XGBoost_RFE": "#2ca02c"}
MAX_DAY = 30

JOBS = [
    ("Analysis/auroc_vs_day",            "AUROC vs. Day — All Balanced, Baseline+Diary"),
    ("Analysis/auroc_vs_day_diary_only", "AUROC vs. Day — All Balanced, Diary Only"),
]

for out_dir, title in JOBS:
    xlsx = os.path.join(out_dir, "auroc_vs_day.xlsx")
    pf = pd.read_excel(xlsx, sheet_name="PerFold")
    summary = (pf.groupby(["Model", "Day"])["Test AUROC"]
                 .agg(["mean", "std", "count"]).reset_index()
                 .rename(columns={"mean": "AUROC_mean", "std": "AUROC_std", "count": "AUROC_n"}))
    summary["AUROC_CI95"] = 1.96 * summary["AUROC_std"] / np.sqrt(summary["AUROC_n"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for mn in [m for m in COLORS if m in summary["Model"].unique()]:
        sub = summary[summary["Model"] == mn].sort_values("Day")
        c = COLORS[mn]
        ax.plot(sub["Day"], sub["AUROC_mean"], "-", color=c, lw=2, label=mn)
        ax.fill_between(sub["Day"], sub["AUROC_mean"] - sub["AUROC_CI95"],
                        sub["AUROC_mean"] + sub["AUROC_CI95"], color=c, alpha=0.15)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="Chance")
    ax.set_xlabel("Day post-surgery", fontsize=12)
    ax.set_ylabel("Test AUROC", fontsize=12)
    ax.set_title(f"{title}\n(mean ± 95% CI, 50 outer folds)", fontweight="bold", fontsize=13)
    ax.set_xlim(1, MAX_DAY); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "auroc_vs_day.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        pf.to_excel(w, sheet_name="PerFold", index=False)
    print(f"✓ {out_dir}/auroc_vs_day.png  (CI band corrected)")

print("Done.")

"""Mann-Whitney — All Balanced, Diary Only. Reads Results/results.xlsx."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from scipy import stats
OUTPUT_DIR = "Analysis/pvalue_diary_only"; ID_COL = "studyid"
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_excel("Results_diary_only/results.xlsx", sheet_name="PerPatient")
models = df["Model"].unique(); results_rows = []
colors_pos = "#d62728"; colors_neg = "#1f77b4"
for model in models:
    sub = df[df["Model"]==model]
    pat = (sub.groupby(ID_COL).agg(proba=("proba","mean"), y_true=("y_true","first"),
            decision_day=("decision_day","mean"), is_fallback=("is_fallback","mean")).reset_index())
    p_pos = pat[pat["y_true"]==1]["proba"].values; p_neg = pat[pat["y_true"]==0]["proba"].values
    stat, pval = stats.mannwhitneyu(p_pos, p_neg, alternative="greater")
    n1,n2 = len(p_pos),len(p_neg); r = (2*stat)/(n1*n2)-1
    sub_pos = pat[pat["y_true"]==1]; sub_neg = pat[pat["y_true"]==0]
    pstr = "p < 0.001" if pval<0.001 else ("p < 0.01" if pval<0.01 else f"p = {pval:.3f}")
    results_rows.append({"Model":model,"N_CPSP1":n1,"N_CPSP0":n2,
        "Median_proba_CPSP1":float(np.median(p_pos)),"IQR_CPSP1":float(np.percentile(p_pos,75)-np.percentile(p_pos,25)),
        "Median_proba_CPSP0":float(np.median(p_neg)),"IQR_CPSP0":float(np.percentile(p_neg,75)-np.percentile(p_neg,25)),
        "Mann-Whitney_U":float(stat),"p_value":float(pval),"p_value_str":pstr,"Effect_size_r":float(r),
        "Pct_fallback_CPSP1":float(sub_pos["is_fallback"].mean()),
        "Pct_fallback_CPSP0":float(sub_neg["is_fallback"].mean()),
        "Mean_dd_CPSP1":float(sub_pos["decision_day"].mean()),
        "Mean_dd_CPSP0":float(sub_neg["decision_day"].mean())})
    print(f"[{model}] U={stat:.0f}, {pstr}, r={r:.3f}")
results_df = pd.DataFrame(results_rows)
# Boxplot
fig, axes = plt.subplots(1,len(models),figsize=(5*len(models),5),sharey=True)
if len(models)==1: axes=[axes]
for ax, model in zip(axes,models):
    sub = df[df["Model"]==model]
    pat = sub.groupby(ID_COL).agg(proba=("proba","mean"),y_true=("y_true","first")).reset_index()
    p_pos=pat[pat["y_true"]==1]["proba"].values; p_neg=pat[pat["y_true"]==0]["proba"].values
    bp = ax.boxplot([p_neg,p_pos],labels=["No CPSP","CPSP"],patch_artist=True,
                    medianprops=dict(color="black",linewidth=2))
    bp["boxes"][0].set_facecolor(colors_neg); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(colors_pos); bp["boxes"][1].set_alpha(0.7)
    row = results_df[results_df["Model"]==model].iloc[0]
    ax.annotate(row["p_value_str"],xy=(1.5,max(p_pos.max(),p_neg.max())*1.02),ha="center",fontsize=10,fontweight="bold")
    ax.set_title(model,fontsize=11,fontweight="bold"); ax.set_ylim(0,1.1)
    ax.axhline(0.5,color="gray",ls="--",lw=1,alpha=0.5)
fig.suptitle("All Balanced — Diary Only\nPredicted probabilities: CPSP=1 vs CPSP=0",fontsize=12,fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,"boxplot_probas.png"),dpi=300,bbox_inches="tight")
plt.close(fig)
# Violin
fig, axes = plt.subplots(1,len(models),figsize=(5*len(models),5),sharey=True)
if len(models)==1: axes=[axes]
for ax, model in zip(axes,models):
    sub = df[df["Model"]==model]
    pat = sub.groupby(ID_COL).agg(proba=("proba","mean"),y_true=("y_true","first")).reset_index()
    p_pos=pat[pat["y_true"]==1]["proba"].values; p_neg=pat[pat["y_true"]==0]["proba"].values
    vp = ax.violinplot([p_neg,p_pos],positions=[1,2],showmedians=True,showextrema=False)
    for body,color in zip(vp["bodies"],[colors_neg,colors_pos]):
        body.set_facecolor(color); body.set_alpha(0.7)
    vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
    row = results_df[results_df["Model"]==model].iloc[0]
    ax.annotate(row["p_value_str"],xy=(1.5,max(p_pos.max(),p_neg.max())*1.02),ha="center",fontsize=10,fontweight="bold")
    ax.set_xticks([1,2]); ax.set_xticklabels(["No CPSP","CPSP"])
    ax.set_title(model,fontsize=11,fontweight="bold"); ax.set_ylim(0,1.15)
fig.suptitle("All Balanced — Diary Only\nPredicted probabilities: CPSP=1 vs CPSP=0",fontsize=12,fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,"violinplot_probas.png"),dpi=300,bbox_inches="tight")
plt.close(fig)
with pd.ExcelWriter(os.path.join(OUTPUT_DIR,"pvalue_results.xlsx"),engine="openpyxl") as w:
    results_df.to_excel(w,sheet_name="MannWhitney",index=False)
print(f"Saved → {OUTPUT_DIR}"); print("Done!")

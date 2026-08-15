"""Regenerate the SHAP beeswarm + bar plots for Figure 6 with human-readable
feature labels (per boss request). Original figures are NOT overwritten — new
files are saved alongside with '_labels' suffix.

Output:
    Best_model_RF/SHAP_beeswarm_labels.png
    Best_model_RF/SHAP_bar_labels.png
    Results/figures/Figure6_combined_labels.png
"""
import os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shap, joblib

T123_PATH = "../STAR_T1_T3_T4_measures_QST_numeric.csv"
T2_PATH   = "../T2_deid_diary_data_numeric.csv"
ID_COL = "studyid"; LABEL_COL = "cpsp_t4_bin"; MAX_DAY = 30

BEST_DIR = "Best_model_RF"
OUT_FIGS = "Results/figures"
os.makedirs(OUT_FIGS, exist_ok=True)

# ── Human-readable labels for each base variable ──
LABEL_MAP = {
    # preop
    "site": "study site",
    "survey_lang": "questionnaire language",
    "gendert1": "sex",
    "child_aget1": "age at surgery",
    "rcads_anxiety_total_tt1": "RCADS anxiety T-score",
    "rcads_mdd_tt1": "RCADS depression T-score",
    "ql_st1": "PedsQL quality of life",
    "pef_st1": "pain self-efficacy (PSEQ)",
    "prghtt1": "PROMIS pediatric global health",
    "prpqtt1": "PROMIS pediatric pain interference",
    "pcq_efat1": "PCQ emotion-focused avoidance",
    "pcq_pfat1": "PCQ problem-focused avoidance",
    "pcq_appt1": "PCQ approach coping",
    "pdc4t1": "baseline pain (NRS)",
    # diary morning
    "amdiary3": "sleep quality (morning)",
    "amd_comp": "morning diary completion",
    "pmd_comp": "evening diary completion",
    # diary pain intensity
    "bpf1": "pain today (y/n)",
    "bpf2": "worst pain (24h)",
    "bpf3": "least pain (24h)",
    "bpf4": "average pain (24h)",
    "bpf5am": "current pain (morning)",
    "bpf5pm": "current pain (evening)",
    # medication
    "bpf6___1": "opioid use",
    "bpf6___2": "acetaminophen use",
    "bpf6___3": "anti-inflammatory use",
    "bpf6___4": "other medication use",
    "bpf6___5": "no medication use",
    "bpf8": "pain relief from medication",
    "opioid_yes": "opioid use (daily flag)",
    "pain_yes": "pain today (daily flag)",
    # pain interference (bpf9-bpf16)
    "bpf9":  "pain interference on general activity",
    "bpf10": "pain interference on mood",
    "bpf11": "pain interference on walking",
    "bpf12": "pain interference on work",
    "bpf13": "pain interference on school work",
    "bpf14": "pain interference on relations",
    "bpf15": "pain interference on sleep",
    "bpf16": "pain interference on enjoyment of life",
}

STAT_RE = re.compile(r'__(last|mean|std|min|max|slope)$')

def prettify(feat):
    """Turn 'bpf10__mean' into 'pain interference on mood_mean'."""
    m = STAT_RE.search(feat)
    if m:
        stat = m.group(1)
        base = feat[:m.start()]
        base_label = LABEL_MAP.get(base, base)
        return f"{base_label}_{stat}"
    else:
        return LABEL_MAP.get(feat, feat)

# ── Rebuild feature matrix identically to RF_fit.py ──
_DAY_RE = re.compile(r'_d(\d{1,2})(?=$|[^0-9])')
def _col_day(c):
    m=_DAY_RE.search(c); return int(m.group(1)) if m else None
def _base_no_day(c): return _DAY_RE.sub('',c).strip('_')
def _safe_slope(d,v):
    m=~np.isnan(v)
    if m.sum()<2: return np.nan
    try: return float(np.polyfit(d[m],v[m],1)[0])
    except: return np.nan

def make_t2_agg(df, day):
    dc=[c for c in df.columns if (dn:=_col_day(c)) is not None and 1<=dn<=day]
    groups={}
    for c in dc: groups.setdefault(_base_no_day(c),[]).append((_col_day(c),c))
    for b in groups: groups[b].sort(key=lambda x:x[0])
    out=pd.DataFrame({ID_COL:df[ID_COL].values})
    for base,d_cols in groups.items():
        if base in {'bpi_pain_mean','bpi_pain_intf','bpi_pain_night'}: continue
        da=np.array([d for d,_ in d_cols],dtype=float)
        vals=df[[c for _,c in d_cols]].to_numpy(dtype=float)
        out[f"{base}__last"]=vals[:,-1]
        out[f"{base}__mean"]=np.nanmean(vals,axis=1)
        out[f"{base}__std"]=np.nanstd(vals,axis=1)
        out[f"{base}__min"]=np.nanmin(vals,axis=1)
        out[f"{base}__max"]=np.nanmax(vals,axis=1)
        out[f"{base}__slope"]=np.apply_along_axis(lambda r:_safe_slope(da,r),1,vals)
    return out

WHITELIST = {
    'site','survey_lang','gendert1','child_aget1',
    'rcads_anxiety_total_tt1','rcads_mdd_tt1',
    'ql_st1','pef_st1','prghtt1','prpqtt1',
    'pcq_efat1','pcq_pfat1','pdc4t1','pcq_appt1',
}

print("Loading data…")
t123=pd.read_csv(T123_PATH); t2=pd.read_csv(T2_PATH)
t123.columns=t123.columns.str.lower(); t2.columns=t2.columns.str.lower()
valid=t123[["pdc4t4","ql_st4"]].notna().all(axis=1)
cond=(t123["pdc4t4"]>=3)&(t123["ql_st4"]<74.9)
t123[LABEL_COL]=np.where(valid,cond.astype(float),np.nan)
t123=t123.loc[t123[LABEL_COL].notna()].copy()
t123[ID_COL]=t123[ID_COL].astype(str); t2[ID_COL]=t2[ID_COL].astype(str)
t123=t123[t123[ID_COL].isin(set(t2[ID_COL]))].copy()
t2=t2[t2[ID_COL].isin(set(t123[ID_COL]))].copy()
ids_all=t123[ID_COL].values
t1_cols=[c for c in t123.columns if c in WHITELIST]
t2_agg=make_t2_agg(t2, MAX_DAY)
t1_part=t123[[ID_COL]+t1_cols].copy()
merged=t1_part.merge(t2_agg,on=ID_COL,how='left')
merged=merged.set_index(ID_COL).select_dtypes(include=np.number)
X=merged.reindex(ids_all)
print(f"N={len(ids_all)}, feature matrix: {X.shape}")

print("Loading fitted pipeline…")
best=joblib.load(os.path.join(BEST_DIR,"best_rf_pipeline.joblib"))
pre=best.named_steps["preprocess"]; rfe=best.named_steps["rfe"]; rf=best.named_steps["model"]
feat_names=np.array([n.replace("num__","") for n in pre.get_feature_names_out()])
sel=feat_names[rfe.support_]
print(f"RFE selected {len(sel)}")

pretty = [prettify(f) for f in sel]
print("Prettified labels:")
for a,b in zip(sel, pretty): print(f"  {a:35s} -> {b}")

X_proc=pre.transform(X); X_rfe=rfe.transform(X_proc)
X_df=pd.DataFrame(X_rfe, columns=pretty)

explainer=shap.TreeExplainer(rf)
sv=explainer.shap_values(X_rfe)
shap_pos = sv[:,:,1] if (isinstance(sv,np.ndarray) and sv.ndim==3) else sv[1]

# ── New beeswarm + bar with pretty labels ──
plt.figure(figsize=(10,10))
shap.summary_plot(shap_pos, X_df, show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(BEST_DIR,"SHAP_beeswarm_labels.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10,10))
shap.summary_plot(shap_pos, X_df, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(BEST_DIR,"SHAP_bar_labels.png"), dpi=300, bbox_inches="tight")
plt.close()

# ── Combined figure ──
img_a=mpimg.imread(os.path.join(BEST_DIR,"SHAP_beeswarm_labels.png"))
img_b=mpimg.imread(os.path.join(BEST_DIR,"SHAP_bar_labels.png"))
fig,(axa,axb)=plt.subplots(1,2, figsize=(13.5,7.0))
for ax,img in zip((axa,axb),(img_a,img_b)):
    ax.imshow(img); ax.axis("off")
for ax,letter in zip((axa,axb),["a","b"]):
    ax.text(0.01,1.01,letter, transform=ax.transAxes, fontsize=20,
            fontweight="bold", va="bottom", ha="left")
fig.subplots_adjust(left=0.01,right=0.99,top=0.97,bottom=0.01,wspace=0.04)
out_combined=os.path.join(OUT_FIGS,"Figure6_combined_labels.png")
fig.savefig(out_combined, dpi=300, bbox_inches="tight")
plt.close(fig)

from PIL import Image
im=Image.open(out_combined)
print(f"\nSaved -> {out_combined}")
print(f"Dimensions: {im.size[0]}x{im.size[1]} px = {im.size[0]/300:.1f}x{im.size[1]/300:.1f} in @300dpi")

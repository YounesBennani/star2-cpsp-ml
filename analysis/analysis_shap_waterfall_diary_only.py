"""SHAP Waterfall — All Balanced, Diary Only. Reads Best_model_RF/."""
import os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap, joblib

OUTPUT_DIR = "Analysis/shap_waterfall_diary_only"; os.makedirs(OUTPUT_DIR, exist_ok=True)
T123_PATH = "../STAR_T1_T3_T4_measures_QST_numeric.csv"
T2_PATH   = "../T2_deid_diary_data_numeric.csv"
ID_COL = "studyid"; LABEL_COL = "cpsp_t4_bin"; MAX_DAY = 30
MODEL_PATH = "Best_model_RF_diary_only/best_rf_pipeline.joblib"

_DAY_RE = re.compile(r'_d(\d{1,2})(?=$|[^0-9])')
def _column_day(col):
    m=_DAY_RE.search(col); return int(m.group(1)) if m else None
def _base_no_day(col): return _DAY_RE.sub('',col).strip('_')
def _safe_slope(d,v):
    m=~np.isnan(v)
    if m.sum()<2: return np.nan
    try: return float(np.polyfit(d[m],v[m],1)[0])
    except: return np.nan

def make_t2_aggregates(df_t2, day):
    dc=[c for c in df_t2.columns if (dn:=_column_day(c)) is not None and 1<=dn<=day]
    if not dc: return pd.DataFrame({ID_COL:df_t2[ID_COL].values})
    groups={}
    for c in dc: groups.setdefault(_base_no_day(c),[]).append((_column_day(c),c))
    for b in groups: groups[b].sort(key=lambda x:x[0])
    out=pd.DataFrame({ID_COL:df_t2[ID_COL].values})
    for base,d_cols in groups.items():
        if base in {'bpi_pain_mean', 'bpi_pain_intf', 'bpi_pain_night'}: continue
        da=np.array([d for d,_ in d_cols],dtype=float)
        vals=df_t2[[c for _,c in d_cols]].to_numpy(dtype=float)
        out[f"{base}__mean"]=np.nanmean(vals,axis=1)
        out[f"{base}__std"]=np.nanstd(vals,axis=1)
    return out

def select_t1_cols(df_t123): return []  # diary-only

if __name__=="__main__":
    print("Loading data..."); t123=pd.read_csv(T123_PATH); t2=pd.read_csv(T2_PATH)
    t123.columns=t123.columns.str.lower(); t2.columns=t2.columns.str.lower()
    valid=t123[["pdc4t4","ql_st4"]].notna().all(axis=1)
    cond=(t123["pdc4t4"]>=3)&(t123["ql_st4"]<74.9)
    t123[LABEL_COL]=np.where(valid,cond.astype(float),np.nan)
    t123=t123.loc[t123[LABEL_COL].notna()].copy()
    t123[ID_COL]=t123[ID_COL].astype(str); t2[ID_COL]=t2[ID_COL].astype(str)
    t123=t123[t123[ID_COL].isin(set(t2[ID_COL]))].copy(); t2=t2[t2[ID_COL].isin(set(t123[ID_COL]))].copy()
    ids_all=t123[ID_COL].values; y=t123[LABEL_COL].astype(int).values
    t1_cols=select_t1_cols(t123)
    t2_agg=make_t2_aggregates(t2,MAX_DAY)
    merged=t123[[ID_COL]+t1_cols].merge(t2_agg,on=ID_COL,how='left')
    merged=merged.set_index(ID_COL).select_dtypes(include=np.number); X=merged.reindex(ids_all)
    print(f"Loading {MODEL_PATH}..."); best_pipe=joblib.load(MODEL_PATH)
    pre=best_pipe.named_steps["preprocess"]; rfe=best_pipe.named_steps["rfe"]; rf=best_pipe.named_steps["model"]
    feat_names=np.array([n.replace("num__","") for n in pre.get_feature_names_out()])
    sel_feat=feat_names[rfe.support_]
    X_proc=pre.transform(X); X_rfe=rfe.transform(X_proc)
    X_rfe_df=pd.DataFrame(X_rfe,columns=sel_feat,index=ids_all)
    probas=rf.predict_proba(X_rfe)[:,1]
    print(f"Selected features: {len(sel_feat)}, proba range: [{probas.min():.3f},{probas.max():.3f}]")
    print("Computing SHAP..."); explainer=shap.TreeExplainer(rf); sv=explainer.shap_values(X_rfe)
    if isinstance(sv,np.ndarray) and sv.ndim==3: shap_pos=sv[:,:,1]; base_val=explainer.expected_value[1]
    else: shap_pos=sv[1]; base_val=explainer.expected_value[1]
    tp_mask=(y==1)&(probas>=0.5); tp_idx=np.where(tp_mask)[0][np.argmax(probas[tp_mask])]
    tn_mask=(y==0)&(probas<0.5);  tn_idx=np.where(tn_mask)[0][np.argmin(probas[tn_mask])]
    border_idx=np.argmin(np.abs(probas-0.5))
    patients=[(tp_idx,"High-risk (True Positive)","high_risk"),
              (tn_idx,"Low-risk (True Negative)","low_risk"),
              (border_idx,"Borderline (closest to 0.5)","borderline")]
    for pid_idx,label,fname in patients:
        pid=ids_all[pid_idx]
        print(f"  {label}: ID={pid}, y={y[pid_idx]}, proba={probas[pid_idx]:.3f}")
        shap_exp=shap.Explanation(values=shap_pos[pid_idx],base_values=base_val,
                                  data=X_rfe_df.iloc[pid_idx].values,feature_names=list(sel_feat))
        fig,ax=plt.subplots(figsize=(10,7)); shap.waterfall_plot(shap_exp,max_display=15,show=False)
        plt.title(f"SHAP Waterfall (All Balanced, Diary Only) — {label}\n"
                  f"Patient {pid} | y_true={y[pid_idx]} | proba={probas[pid_idx]:.3f}",fontsize=12,fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,f"waterfall_{fname}.png"),dpi=300,bbox_inches="tight")
        plt.close()
    fig,ax=plt.subplots(figsize=(10,7))
    shap.decision_plot(base_val,shap_pos,X_rfe_df,feature_order="importance",highlight=None,show=False)
    plt.title("SHAP Decision Plot — All Balanced, Diary Only",fontsize=13,fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"decision_plot.png"),dpi=300,bbox_inches="tight"); plt.close()
    rows=[{"Patient_ID":ids_all[i],"Profile":lbl,"y_true":int(y[i]),"Predicted_proba":float(probas[i]),
           **{f"SHAP_{f}":float(shap_pos[i,j]) for j,f in enumerate(sel_feat)}}
          for i,lbl,_ in patients]
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR,"waterfall_patients.xlsx"),engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w,sheet_name="Selected_Patients",index=False)
    print(f"All outputs → {OUTPUT_DIR}/\nDone!")

"""RF full-dataset fit + SHAP — All Balanced, Diary Only. Output → Best_model_RF/"""
import os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap, joblib
from typing import Dict, List, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

T123_PATH = "../STAR_T1_T3_T4_measures_QST_numeric.csv"
T2_PATH   = "../T2_deid_diary_data_numeric.csv"
ID_COL = "studyid"; LABEL_COL = "cpsp_t4_bin"; MAX_DAY = 30; RANDOM_STATE = 42
BEST_MODEL_DIR = "Best_model_RF_diary_only"
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

param_grid_rf = {
    "rfe__n_features_to_select": [10,15,20],
    "model__n_estimators":       [300],
    "model__max_depth":          [2,3,4],
    "model__min_samples_leaf":   [10,15,20],
    "model__max_features":       ["sqrt",0.3],
    "model__class_weight":       ["balanced"],
}

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
    print("Loading data...")
    t123=pd.read_csv(T123_PATH); t2=pd.read_csv(T2_PATH)
    t123.columns=t123.columns.str.lower(); t2.columns=t2.columns.str.lower()
    valid=t123[["pdc4t4","ql_st4"]].notna().all(axis=1)
    cond=(t123["pdc4t4"]>=3)&(t123["ql_st4"]<74.9)
    t123[LABEL_COL]=np.where(valid,cond.astype(float),np.nan)
    t123=t123.loc[t123[LABEL_COL].notna()].copy()
    t123[ID_COL]=t123[ID_COL].astype(str); t2[ID_COL]=t2[ID_COL].astype(str)
    t123=t123[t123[ID_COL].isin(set(t2[ID_COL]))].copy()
    t2=t2[t2[ID_COL].isin(set(t123[ID_COL]))].copy()
    ids_all=t123[ID_COL].values; y=t123[LABEL_COL].astype(int).values
    t1_cols=select_t1_cols(t123)
    print(f"N={len(ids_all)}, T1 feats={len(t1_cols)}")
    t2_agg=make_t2_aggregates(t2,MAX_DAY)
    t1_part=t123[[ID_COL]+t1_cols].copy()
    merged=t1_part.merge(t2_agg,on=ID_COL,how='left')
    merged=merged.set_index(ID_COL).select_dtypes(include=np.number)
    X=merged.reindex(ids_all); print(f"Feature matrix: {X.shape}")
    preprocess=ColumnTransformer([("num",SimpleImputer(strategy="median"),
                                   make_column_selector(dtype_include=np.number))],remainder="drop")
    pipeline=Pipeline([("preprocess",preprocess),
                       ("rfe",RFE(RandomForestClassifier(n_estimators=100,random_state=RANDOM_STATE),step=0.3)),
                       ("model",RandomForestClassifier(random_state=RANDOM_STATE))])
    search=GridSearchCV(pipeline,param_grid_rf,
                        cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=RANDOM_STATE),
                        scoring="roc_auc",refit=True,n_jobs=8,verbose=1,return_train_score=True)
    search.fit(X,y)
    print(f"Best params: {search.best_params_}"); print(f"Best CV AUROC: {search.best_score_:.4f}")
    best_pipe=search.best_estimator_
    pre=best_pipe.named_steps["preprocess"]; rfe=best_pipe.named_steps["rfe"]; rf=best_pipe.named_steps["model"]
    feat_names=np.array([n.replace("num__","") for n in pre.get_feature_names_out()])
    sel_feat=feat_names[rfe.support_]
    print(f"RFE selected {len(sel_feat)} features: {list(sel_feat)}")
    X_proc=pre.transform(X); X_rfe=rfe.transform(X_proc); X_rfe_df=pd.DataFrame(X_rfe,columns=sel_feat)
    explainer=shap.TreeExplainer(rf); sv=explainer.shap_values(X_rfe)
    shap_pos=sv[:,:,1] if (isinstance(sv,np.ndarray) and sv.ndim==3) else sv[1]
    for plot_type,fname in [("beeswarm","SHAP_beeswarm"),("bar","SHAP_bar")]:
        plt.figure(figsize=(10,8))
        if plot_type=="beeswarm": shap.summary_plot(shap_pos,X_rfe_df,show=False)
        else: shap.summary_plot(shap_pos,X_rfe_df,plot_type="bar",show=False)
        plt.title(f"SHAP {plot_type.capitalize()} — RF All Balanced, Diary Only",fontsize=13,fontweight="bold")
        plt.tight_layout(); plt.savefig(os.path.join(BEST_MODEL_DIR,f"{fname}.png"),dpi=300,bbox_inches="tight"); plt.close()
    mean_abs=np.abs(shap_pos).mean(axis=0); feat_sum=pd.DataFrame({
        "Feature":sel_feat,"Mean_|SHAP|":mean_abs,"Std_|SHAP|":np.abs(shap_pos).std(axis=0),
        "Mean_SHAP_signed":shap_pos.mean(axis=0),"Gini_Importance":rf.feature_importances_
    }).sort_values("Mean_|SHAP|",ascending=False).reset_index(drop=True)
    feat_sum["SHAP_Rank"]=range(1,len(feat_sum)+1)
    joblib.dump(best_pipe,os.path.join(BEST_MODEL_DIR,"best_rf_pipeline.joblib"))
    cv_df=pd.DataFrame(search.cv_results_)[["params","mean_test_score","std_test_score","rank_test_score"]].sort_values("rank_test_score")
    rfe_df=pd.DataFrame({"Feature":feat_names,"RFE_Ranking":rfe.ranking_,"Selected":rfe.support_}).sort_values("RFE_Ranking")
    with pd.ExcelWriter(os.path.join(BEST_MODEL_DIR,"RF_final_results.xlsx"),engine="openpyxl") as w:
        pd.DataFrame([search.best_params_]).to_excel(w,sheet_name="Best_Params",index=False)
        cv_df.to_excel(w,sheet_name="CV_Results",index=False)
        rfe_df.to_excel(w,sheet_name="RFE_All_Features",index=False)
        feat_sum.to_excel(w,sheet_name="Feature_Importance",index=False)
    print(f"All outputs → {BEST_MODEL_DIR}/\nDone!")

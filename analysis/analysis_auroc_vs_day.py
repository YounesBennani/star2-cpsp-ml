"""AUROC vs Day — All Balanced, Baseline+Diary. Output → Analysis/auroc_vs_day/"""
import os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

T123_PATH = "../STAR_T1_T3_T4_measures_QST_numeric.csv"
T2_PATH   = "../T2_deid_diary_data_numeric.csv"
ID_COL = "studyid"; LABEL_COL = "cpsp_t4_bin"; MAX_DAY = 30
RANDOM_STATE = 42; OUTER_SPLITS = 5; N_REPEATS = 10; INNER_SPLITS = 5; DEV_SIZE = 0.20
SCALE_POS_WEIGHT = round(97/47, 4)
OUTPUT_DIR = "Analysis/auroc_vs_day"; os.makedirs(OUTPUT_DIR, exist_ok=True)

param_grid_lr  = {"model__penalty":["elasticnet"],"model__solver":["saga"],
                  "model__l1_ratio":[0.1,0.5,0.9,1.0],"model__C":list(np.logspace(-4,-1,5)),
                  "model__class_weight":["balanced"]}
param_grid_rf  = {"rfe__n_features_to_select":[10,15,20],"model__n_estimators":[300],
                  "model__max_depth":[2,3,4],"model__min_samples_leaf":[10,15,20],
                  "model__max_features":["sqrt",0.3],"model__class_weight":["balanced"]}
param_grid_xgb = {"rfe__n_features_to_select":[10,15,20],"model__n_estimators":[200],
                  "model__learning_rate":[0.03,0.1],"model__max_depth":[2,3],
                  "model__subsample":[0.8],"model__colsample_bytree":[0.8],
                  "model__min_child_weight":[5,10],"model__reg_alpha":[0.1,0.5],
                  "model__reg_lambda":[1,5,10],"model__scale_pos_weight":[SCALE_POS_WEIGHT]}

_DAY_RE = re.compile(r'_d(\d{1,2})(?=$|[^0-9])')
def _column_day(col):
    m=_DAY_RE.search(col); return int(m.group(1)) if m else None
def _base_no_day(col): return _DAY_RE.sub('',col).strip('_')
def _safe_slope(d,v):
    m=~np.isnan(v)
    if m.sum()<2: return np.nan
    try: return float(np.polyfit(d[m],v[m],1)[0])
    except: return np.nan
def make_t2_aggregates(df_t2,day):
    dc=[c for c in df_t2.columns if (dn:=_column_day(c)) is not None and 1<=dn<=day]
    if not dc: return pd.DataFrame({ID_COL:df_t2[ID_COL].values})
    groups={}
    for c in dc: groups.setdefault(_base_no_day(c),[]).append((_column_day(c),c))
    for b in groups: groups[b].sort(key=lambda x:x[0])
    out=pd.DataFrame({ID_COL:df_t2[ID_COL].values})
    for base,d_cols in groups.items():
        if base in {'bpi_pain_mean', 'bpi_pain_intf', 'bpi_pain_night'}: continue
        da=np.array([d for d,_ in d_cols],dtype=float); vals=df_t2[[c for _,c in d_cols]].to_numpy(dtype=float)
        out[f"{base}__mean"]=np.nanmean(vals,axis=1)
        out[f"{base}__std"]=np.nanstd(vals,axis=1)
    return out
def select_t1_cols(df_t123):
    WHITELIST = {
        'site', 'survey_lang', 'gendert1', 'child_aget1',
        'rcads_anxiety_total_tt1', 'rcads_mdd_tt1',
        'ql_st1', 'pef_st1', 'prghtt1', 'prpqtt1',
        'pcq_efat1', 'pcq_pfat1', 'pdc4t1', 'pcq_appt1',
    }
    return [c for c in df_t123.columns if c in WHITELIST]
def build_X(df_t123,t2_cache,ids,t1_cols,day,ref_cols=None):
    ids_str=[str(i) for i in ids]; ids_set=set(ids_str)
    t1_part=df_t123[df_t123[ID_COL].isin(ids_set)][[ID_COL]+t1_cols].copy()
    t2_sub=t2_cache[day][t2_cache[day][ID_COL].isin(ids_set)].copy()
    merged=t1_part.merge(t2_sub,on=ID_COL,how='left')
    merged=merged.set_index(ID_COL).select_dtypes(include=np.number).reindex(ids_str)
    if ref_cols is not None: merged=merged.reindex(columns=ref_cols,fill_value=np.nan)
    return merged

preprocess_lr=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),
                                   make_column_selector(dtype_include=np.number))],remainder="drop")
preprocess_tree=ColumnTransformer([("num",SimpleImputer(strategy="median"),
                                    make_column_selector(dtype_include=np.number))],remainder="drop")
models_to_run=[
    {"name":"LogReg_ElasticNet","pipeline":Pipeline([("preprocess",preprocess_lr),
     ("model",LogisticRegression(random_state=RANDOM_STATE,max_iter=10000))]),"grid":param_grid_lr},
    {"name":"RandomForest_RFE","pipeline":Pipeline([("preprocess",preprocess_tree),
     ("rfe",RFE(RandomForestClassifier(n_estimators=100,random_state=RANDOM_STATE),step=0.3)),
     ("model",RandomForestClassifier(random_state=RANDOM_STATE))]),"grid":param_grid_rf},
    {"name":"XGBoost_RFE","pipeline":Pipeline([("preprocess",preprocess_tree),
     ("rfe",RFE(XGBClassifier(n_estimators=100,eval_metric="logloss",random_state=RANDOM_STATE),step=0.3)),
     ("model",XGBClassifier(eval_metric="logloss",random_state=RANDOM_STATE))]),"grid":param_grid_xgb},
]

if __name__=="__main__":
    print("Loading data..."); t123=pd.read_csv(T123_PATH); t2=pd.read_csv(T2_PATH)
    t123.columns=t123.columns.str.lower(); t2.columns=t2.columns.str.lower()
    valid=t123[["pdc4t4","ql_st4"]].notna().all(axis=1); cond=(t123["pdc4t4"]>=3)&(t123["ql_st4"]<74.9)
    t123[LABEL_COL]=np.where(valid,cond.astype(float),np.nan); t123=t123.loc[t123[LABEL_COL].notna()].copy()
    t123[ID_COL]=t123[ID_COL].astype(str); t2[ID_COL]=t2[ID_COL].astype(str)
    t123=t123[t123[ID_COL].isin(set(t2[ID_COL]))].copy(); t2=t2[t2[ID_COL].isin(set(t123[ID_COL]))].copy()
    ids_all=t123[ID_COL].values; y_all=t123[LABEL_COL].astype(int).values; t1_cols=select_t1_cols(t123)
    print(f"N={len(ids_all)}, T1 feats={len(t1_cols)}")
    t2_cache={}
    for d in range(1,MAX_DAY+1): t2_cache[d]=make_t2_aggregates(t2,d)
    outer_cv=RepeatedStratifiedKFold(n_splits=OUTER_SPLITS,n_repeats=N_REPEATS,random_state=RANDOM_STATE)
    inner_cv=RepeatedStratifiedKFold(n_splits=INNER_SPLITS,n_repeats=1,random_state=RANDOM_STATE)
    auroc_rows=[]
    for m in models_to_run:
        print(f"\n>>> {m['name']}")
        for idx,(tr_idx,te_idx) in enumerate(outer_cv.split(ids_all,y_all)):
            rep=idx//OUTER_SPLITS+1; fold=idx%OUTER_SPLITS+1
            train_ids=ids_all[tr_idx]; test_ids=ids_all[te_idx]
            y_train=y_all[tr_idx]; y_test=y_all[te_idx]
            itr_ids,_,y_itr,_=train_test_split(train_ids,y_train,test_size=DEV_SIZE,
                                                stratify=y_train,random_state=RANDOM_STATE+idx)
            X_max=build_X(t123,t2_cache,itr_ids,t1_cols,MAX_DAY); ref_cols=list(X_max.columns)
            search=GridSearchCV(m["pipeline"],m["grid"],cv=inner_cv,scoring="roc_auc",refit=True,n_jobs=8)
            search.fit(X_max,y_itr); best_pipe=search.best_estimator_
            for d in range(1,MAX_DAY+1):
                X_test_d=build_X(t123,t2_cache,test_ids,t1_cols,d,ref_cols)
                try: auroc=float(roc_auc_score(y_test,best_pipe.predict_proba(X_test_d)[:,1]))
                except: auroc=np.nan
                auroc_rows.append({"Model":m["name"],"Repeat":rep,"Fold":fold,"Day":d,"Test AUROC":auroc})
            if (idx+1)%10==0: print(f"  {idx+1}/50 done")
    auroc_df=pd.DataFrame(auroc_rows)
    summary=auroc_df.groupby(["Model","Day"])["Test AUROC"].agg(["mean","std","count"]).reset_index().rename(columns={"mean":"AUROC_mean","std":"AUROC_std","count":"AUROC_n"})
    colors={"LogReg_ElasticNet":"#1f77b4","RandomForest_RFE":"#ff7f0e","XGBoost_RFE":"#2ca02c"}
    fig,ax=plt.subplots(figsize=(10,6))
    for mn in [m["name"] for m in models_to_run]:
        sub=summary[summary["Model"]==mn].sort_values("Day"); c=colors.get(mn,"gray")
        ci=1.96*sub["AUROC_std"]/np.sqrt(sub["AUROC_n"])
        ax.plot(sub["Day"],sub["AUROC_mean"],"-",color=c,lw=2,label=mn)
        ax.fill_between(sub["Day"],sub["AUROC_mean"]-ci,sub["AUROC_mean"]+ci,color=c,alpha=0.15)
    ax.axhline(0.5,color="k",ls="--",lw=1,label="Chance")
    ax.set_xlabel("Day post-surgery",fontsize=12); ax.set_ylabel("Test AUROC",fontsize=12)
    ax.set_title("AUROC vs. Day — All Balanced, Baseline+Diary\n(mean ± 95% CI, 50 outer folds)",fontweight="bold",fontsize=13)
    ax.set_xlim(1,MAX_DAY); ax.set_xticks([1,5,10,15,20,25,30]); ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"auroc_vs_day.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR,"auroc_vs_day.xlsx"),engine="openpyxl") as w:
        summary.to_excel(w,sheet_name="Summary",index=False); auroc_df.to_excel(w,sheet_name="PerFold",index=False)
    print(f"Saved → {OUTPUT_DIR}\nDone!")

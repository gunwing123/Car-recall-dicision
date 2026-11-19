import pandas as pd, numpy as np, sys, lightgbm as lgb, json
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d=sys.argv[1]; HZ=12; WIN=3
def norm(df):
    for k in ["make","year"]: df[k]=df[k].str.strip().str.upper()
    df["model"]=df["model"].str.upper().str.replace(r"[^A-Z0-9]","",regex=True)
    df["comp"]=df["comp"].str.split(":").str[0].str.strip().str.upper()
r=pd.read_csv(f"{d}/FLAT_RCL_POST_2010.txt",sep="\t",header=None,usecols=[1,2,3,4,6,10,11,13,15],
   names=["campno","make","model","year","comp","type","potaff","infl","rcdate"],dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
r=r[r.type=="V"]; norm(r); r=r[r.year.str.fullmatch(r"(19|20)\d\d")]
r["dt"]=pd.to_datetime(r.rcdate,format="%Y%m%d",errors="coerce"); r=r.dropna(subset=["dt"])
r["m"]=r.dt.dt.year*12+r.dt.dt.month-1; r["year"]=r.year.astype(int)
r["potaff"]=pd.to_numeric(r.potaff,errors="coerce")
r=r.drop_duplicates(["campno","make","model","year","comp"])
c=pd.read_csv(f"{d}/FLAT_CMPL.txt",sep="\t",header=None,usecols=[3,4,5,6,8,9,11,15],names=["make","model","year","crash","fire","inj","comp","datea"],
   dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
norm(c); c=c[c.year.str.fullmatch(r"(19|20)\d\d")]; c["year"]=c.year.astype(int)
c["dt"]=pd.to_datetime(c.datea,format="%Y%m%d",errors="coerce"); c=c.dropna(subset=["dt"]); c=c[c.dt>="2007-01-01"]
c["m"]=c.dt.dt.year*12+c.dt.dt.month-1
c["sev"]=((c.crash=="Y")|(c.fire=="Y")|(pd.to_numeric(c.inj,errors="coerce").fillna(0)>0)).astype(int)
cm=c.groupby(["make","model","comp","year","m"]).agg(n=("sev","size"),sev=("sev","sum")).reset_index()
# events: campaign x make-model-comp
ev=r.groupby(["campno","make","model","comp"]).agg(m0=("m","min"),ymin=("year","min"),ymax=("year","max"),ny=("year","nunique"),
   years=("year",lambda s:set(s)),potaff=("potaff","max"),infl=("infl","first")).reset_index()
# model-year universe per make-model (seen anywhere)
uni=pd.concat([r[["make","model","year"]],c[["make","model","year"]]]).drop_duplicates()
U=uni.groupby(["make","model"]).year.apply(set).to_dict()
rows=[]
for e in ev.itertuples(index=False):
    for y in U.get((e.make,e.model),()):
        if y in e.years or y<e.ymin-WIN or y>e.ymax+WIN or y>e.m0//12+1: continue
        rows.append((e.campno,e.make,e.model,e.comp,e.m0,y,e.ymin,e.ymax,e.ny,e.potaff,e.infl))
C=pd.DataFrame(rows,columns=["campno","make","model","comp","m0","cy","ymin","ymax","ny","potaff","infl"])
C["dist"]=np.where(C.cy<C.ymin,C.ymin-C.cy,C.cy-C.ymax)
# label: a DIFFERENT campaign recalls (make,model,comp,cy) within (m0, m0+HZ]
rl=r[["campno","make","model","comp","year","m"]].rename(columns={"year":"cy","campno":"camp2","m":"m2"})
L=C.merge(rl,on=["make","model","comp","cy"]); L=L[(L.camp2!=L.campno)]
fut=L[(L.m2>L.m0)&(L.m2<=L.m0+HZ)].groupby(["campno","make","model","comp","cy"]).size().rename("y")
past=L[(L.m2<=L.m0)&(L.m2>L.m0-24)].groupby(["campno","make","model","comp","cy"]).size().rename("cand_prior_rcl24")
C=C.join(fut,on=["campno","make","model","comp","cy"]).join(past,on=["campno","make","model","comp","cy"])
C["y"]=C.y.notna().astype(int); C["cand_prior_rcl24"]=C.cand_prior_rcl24.fillna(0)
# complaint features up to m0 (inclusive)
def agg(keys,yearcol,pref,yearsel=None):
    pass
cmk=cm.set_index(["make","model","comp"])
def cfeat(df):
    out=[]
    g=cm.merge(df[["campno","make","model","comp","m0","cy","ymin","ymax"]].drop_duplicates(),on=["make","model","comp"])
    g=g[g.m<=g.m0]; g["lag"]=g.m0-g.m
    cand=g[g.year==g.cy]; rec=g[(g.year>=g.ymin)&(g.year<=g.ymax)]
    k=["campno","make","model","comp","cy"]
    f=pd.DataFrame(index=df.set_index(k).index)
    for nm,s in [("cand",cand),("rec",rec)]:
        for w in [3,12,24]:
            f[f"{nm}_n{w}"]=s[s.lag<w].groupby(k).n.sum()
        f[f"{nm}_sev12"]=s[s.lag<12].groupby(k).sev.sum()
        f[f"{nm}_ncum"]=s.groupby(k).n.sum()
    return f.fillna(0).reset_index()
F=cfeat(C); C=C.merge(F,on=["campno","make","model","comp","cy"])
C["rec_n12_per_year"]=C.rec_n12/C.ny
C["cand_vs_rec"]=(C.cand_n12+0.5)/(C.rec_n12_per_year+0.5)
C["cand_growth"]=(C.cand_n3/3+0.5)/(C.cand_n12/12+0.5)
C["cand_sev_rate"]=C.cand_sev12/C.cand_n12.clip(lower=1)
C["side_after"]=(C.cy>C.ymax).astype(int); C["age_cand"]=C.m0//12-C.cy
C["infl_odi"]=(C.infl.str.strip()!="MFR").astype(int); C["log_potaff"]=np.log1p(C.potaff.fillna(0))
C["ev_year"]=C.m0//12
C.to_pickle(f"{d}/expansion.pkl")
print("events (campaign x make-model-comp):",len(ev)," candidates:",len(C)," positives:",int(C.y.sum()),f"base rate {C.y.mean():.3%}")
ev2=C.groupby("campno").y.max(); print(f"campaigns with >=1 later expansion within {HZ}m: {ev2.mean():.2%} of {len(ev2)}")
print("base rate by distance:",C.groupby("dist").y.mean().round(4).to_dict())
print("base rate by ODI vs MFR:",C.groupby("infl_odi").y.mean().round(4).to_dict())
print("lag to expansion (months, median):",float((L[(L.m2>L.m0)&(L.m2<=L.m0+HZ)].groupby(["campno","cy"]).m2.min()-L[(L.m2>L.m0)&(L.m2<=L.m0+HZ)].groupby(["campno","cy"]).m0.first()).median()))
print(C.groupby("ev_year").y.agg(["size","mean"]).round(4).T.to_string())
# ---- model
feats=["dist","side_after","age_cand","ny","log_potaff","infl_odi","cand_prior_rcl24","cand_n3","cand_n12","cand_n24","cand_sev12","cand_ncum",
       "rec_n3","rec_n12","rec_n24","rec_sev12","rec_ncum","rec_n12_per_year","cand_vs_rec","cand_growth","cand_sev_rate"]
C["comp_c"]=C.comp.astype("category")
tr=C.ev_year<=2018; va=C.ev_year.between(2019,2020); te=C.ev_year.between(2021,2025)
print({k:(int(s.sum()),int(C.y[s].sum()),round(C.y[s].mean(),4)) for k,s in [("train",tr),("valid",va),("test",te)]})
T=C[te]; res={}
def ev_(name,s):
    o={"PR-AUC":round(average_precision_score(T.y,s),4),"ROC-AUC":round(roc_auc_score(T.y,s),4)}
    for q in [0.05,0.10,0.20]:
        k=int(len(T)*q); idx=np.argsort(-s)[:k]
        o[f"precision@top{int(q*100)}%"]=round(T.y.values[idx].mean(),3); o[f"recall@top{int(q*100)}%"]=round(T.y.values[idx].sum()/T.y.sum(),3)
    res[name]=o
ev_("rule: dist=1 & cand complaints",(-T.dist+T.cand_n12/1e4).values)
def lx(D): 
    X=D[feats].copy()
    for f in feats:
        if f not in ["dist","side_after","age_cand","infl_odi","log_potaff","cand_sev_rate","cand_vs_rec","cand_growth"]: X[f]=np.log1p(X[f])
    X["cand_vs_rec"]=np.log(X.cand_vs_rec); X["cand_growth"]=np.log(X.cand_growth)
    return pd.concat([X,pd.get_dummies(D.comp,prefix="c",dtype=float)],axis=1)
Xtr=lx(C[tr]); cols=Xtr.columns; sc=StandardScaler().fit(Xtr)
lr=LogisticRegression(max_iter=500,C=0.5).fit(sc.transform(Xtr),C.y[tr])
ev_("logistic",lr.predict_proba(sc.transform(lx(T).reindex(columns=cols,fill_value=0)))[:,1])
ff=feats+["comp_c"]
prm=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,
         lambda_l2=10,cat_smooth=50,min_data_per_group=200,verbose=-1,seed=0,num_threads=4)
m=lgb.train(prm,lgb.Dataset(C.loc[tr,ff],C.y[tr]),2000,valid_sets=[lgb.Dataset(C.loc[va,ff],C.y[va])],callbacks=[lgb.early_stopping(150,verbose=False)])
print("lgb best iter",m.best_iteration)
ev_("lightgbm",m.predict(T[ff],num_iteration=m.best_iteration))
print(pd.DataFrame(res).T.to_string())
imp=pd.Series(m.feature_importance("gain"),index=ff).sort_values(ascending=False); print((imp/imp.sum()).round(3).head(10).to_string())
json.dump(res,open(f"{d}/expansion_results.json","w"),ensure_ascii=False,indent=1)

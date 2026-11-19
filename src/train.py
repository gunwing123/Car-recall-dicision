import pandas as pd, numpy as np, sys, time, lightgbm as lgb, json
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d=sys.argv[1]; t0=time.time()
P=pd.read_pickle(f"{d}/panel.pkl")
eps=0.5
P["growth3_12"]=(P.n3/3+eps)/(P.n12/12+eps)
P["growth6_prev6"]=(P.n6+eps)/(P.n_prev6+eps)
P["growth12_prev12"]=(P.n12+eps)/(P.n_prev12+eps)
for s in ["crash","fire","inj","dth"]: P[f"{s}_rate12"]=P[f"{s}12"]/P.n12
P["comp_c"]=P.comp.astype("category")
top=P.make.value_counts().index[:40]
P["make_c"]=P.make.where(P.make.isin(top),"OTHER").astype("category")
num=["n","n3","n6","n12","n24","n_prev6","n_prev12","n_cum","crash12","fire12","inj12","dth12",
     "crash_rate12","fire_rate12","inj_rate12","dth_rate12","growth3_12","growth6_prev6","growth12_prev12",
     "mon_since_first","age","prior_rcl","sib_n12","mc_n12","mmc_rcl12"]
cat=["comp_c","make_c"]
Y=lambda a,b:(a)*12+b-1
tr=P.m<=Y(2019,12); va=(P.m>=Y(2020,1))&(P.m<=Y(2021,12)); te=P.m>=Y(2022,1)
print({k:(int(s.sum()),int(P.y[s].sum())) for k,s in [("train",tr),("valid",va),("test",te)]})

scores={}
T=P[te].copy()
# 1) rule: recent complaint volume
scores["rule_n3"]=(T.n3+T.n12/1000).values
# 2) logistic regression
def X_lr(D):
    X=np.log1p(D[[c for c in num if not c.startswith("growth") and "rate" not in c]].clip(lower=0))
    X=pd.concat([X,np.log(D[[c for c in num if c.startswith("growth")]]),D[[c for c in num if "rate" in c]],
                 pd.get_dummies(D.comp_c,prefix="c",dtype=np.float32)],axis=1)
    return X.astype(np.float32)
rs=np.random.default_rng(0)
idx=P.index[tr]; pos=idx[P.y[tr].values==1]; neg=rs.choice(idx[P.y[tr].values==0],800000,replace=False)
S=P.loc[np.concatenate([pos,neg])]
Xs=X_lr(S); cols=Xs.columns; sc=StandardScaler().fit(Xs)
lr=LogisticRegression(max_iter=300,C=0.5).fit(sc.transform(Xs),S.y)
print("logistic valid AP",round(average_precision_score(P.y[va],lr.predict_proba(sc.transform(X_lr(P[va]).reindex(columns=cols,fill_value=0)))[:,1]),4))
scores["logistic"]=lr.predict_proba(sc.transform(X_lr(T).reindex(columns=cols,fill_value=0)))[:,1]
print(f"logistic done {time.time()-t0:.0f}s")
# 3) LightGBM (+ ablation without trend/severity features)
def fit(feats,name):
    dtr=lgb.Dataset(P.loc[tr,feats],P.y[tr],categorical_feature=[c for c in feats if c in cat])
    dva=lgb.Dataset(P.loc[va,feats],P.y[va],reference=dtr)
    prm=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_data_in_leaf=2000,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,
             lambda_l2=10,cat_smooth=50,cat_l2=50,min_data_per_group=2000,verbose=-1,seed=0,num_threads=4)
    m=lgb.train(prm,dtr,1500,valid_sets=[dva],callbacks=[lgb.early_stopping(150,verbose=False)])
    print(name,"best_iter",m.best_iteration,"valid AP",round(average_precision_score(P.y[va],m.predict(P.loc[va,feats],num_iteration=m.best_iteration)),4),f"{time.time()-t0:.0f}s")
    return m
vol=["n","n3","n6","n12","n24","n_cum","mon_since_first","age","comp_c"]
m_vol=fit(vol,"lgb_volume_only"); scores["lgb_volume_only"]=m_vol.predict(T[vol],num_iteration=m_vol.best_iteration)
full=num+["comp_c"]
m=fit(full,"lgb_full"); scores["lgb_full"]=m.predict(T[full],num_iteration=m.best_iteration)
m.save_model(f"{d}/lgb_full.txt")

# evaluation
rk=pd.read_pickle(f"{d}/recall_units.pkl")
res={}
T=T[["kid","m","y","comp"]].copy()
for name,s in scores.items():
    T["s"]=s
    ap=average_precision_score(T.y,s); auc=roc_auc_score(T.y,s)
    out={"PR-AUC":round(ap,4),"ROC-AUC":round(auc,4)}
    T["rank"]=T.groupby("m").s.rank(ascending=False,method="first")
    for K in [100,500]:
        top=T[T["rank"]<=K]
        mon=T.groupby("m").y.sum()
        hit=top.groupby("m").y.sum().reindex(mon.index,fill_value=0)
        out[f"P@{K}"]=round((hit/K).mean(),4); out[f"R@{K}"]=round(hit.sum()/mon.sum(),4)
        # recall-unit level: flagged in any of the 6 months before recall month
        flag=top[["kid","m"]]
        U=rk[(rk.m>=Y(2022,2))&(rk.m<=Y(2026,9))][["kid","m"]].drop_duplicates().rename(columns={"m":"rm"})
        eligible=U.merge(T[T.y==1][["kid","m"]],on="kid")
        eligible=eligible[(eligible.rm-eligible.m).between(1,6)][["kid","rm"]].drop_duplicates()
        f=eligible.merge(flag,on="kid"); f=f[(f.rm-f.m).between(1,6)]
        caught=f.groupby(["kid","rm"]).m.min()
        out[f"units_caught@{K}"]=f"{len(caught)}/{len(eligible)} ({len(caught)/len(eligible):.1%})"
        out[f"median_lead@{K}"]=float(np.median(caught.index.get_level_values(1)-caught.values)) if len(caught) else None
    res[name]=out
print(pd.DataFrame(res).T.to_string())
json.dump(res,open(f"{d}/results.json","w"),ensure_ascii=False,indent=1)
imp=pd.Series(m.feature_importance("gain"),index=full).sort_values(ascending=False)
print((imp/imp.sum()).round(3).head(15).to_string())
imp.to_csv(f"{d}/importance.csv")
print("test positive rate",round(T.y.mean(),4),"rows/month",int(T.groupby("m").size().mean()))

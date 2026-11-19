import pandas as pd, numpy as np, sys, json
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d=sys.argv[1]; Y=lambda a,b:a*12+b-1
P=pd.read_pickle(f"{d}/panel.pkl"); R=pd.read_pickle(f"{d}/text_feats.pkl")
P=P.merge(R,on=["kid","m"],how="left"); del R
eps=0.5
P["growth3_12"]=(P.n3/3+eps)/(P.n12/12+eps); P["growth6_prev6"]=(P.n6+eps)/(P.n_prev6+eps); P["growth12_prev12"]=(P.n12+eps)/(P.n_prev12+eps)
for s in ["crash","fire","inj","dth"]: P[f"{s}_rate12"]=P[f"{s}12"]/P.n12
cnt=["n","n3","n6","n12","n24","n_prev6","n_prev12","n_cum","crash12","fire12","inj12","dth12","mon_since_first","age","prior_rcl","sib_n12","mc_n12","mmc_rcl12"]
grw=["growth3_12","growth6_prev6","growth12_prev12"]; rate=["crash_rate12","fire_rate12","inj_rate12","dth_rate12"]
kw=[c for c in P.columns if c.startswith("kw_")]; tp=[c for c in P.columns if c.startswith("tp") and c!="tp02_r12"]
P[kw+tp+["tp02_r12","mention_recall_r12"]]=P[kw+tp+["tp02_r12","mention_recall_r12"]].fillna(0)
tr=P.m<=Y(2019,12); va=(P.m>=Y(2020,1))&(P.m<=Y(2021,12)); te=P.m>=Y(2022,1)
def X(D,extra):
    Z=pd.concat([np.log1p(D[cnt].clip(lower=0)),np.log(D[grw]),D[rate+extra],pd.get_dummies(D.comp,prefix="c",dtype=np.float32)],axis=1)
    return Z.astype(np.float32)
rs=np.random.default_rng(0); idx=P.index[tr]
S=P.loc[np.concatenate([idx[P.y[tr].values==1],rs.choice(idx[P.y[tr].values==0],800000,replace=False)])]
T=P[te]; res={}
for name,extra in [("base (no text)",[]),("+ keywords",kw),("+ keywords + topics",kw+tp),("+ recall-related text (leaky check)",kw+tp+["tp02_r12","mention_recall_r12"])]:
    Xs=X(S,extra); cols=Xs.columns; sc=StandardScaler().fit(Xs)
    lr=LogisticRegression(max_iter=400,C=0.5).fit(sc.transform(Xs),S.y)
    pv=lr.predict_proba(sc.transform(X(P[va],extra).reindex(columns=cols,fill_value=0)))[:,1]
    s=lr.predict_proba(sc.transform(X(T,extra).reindex(columns=cols,fill_value=0)))[:,1]
    o={"valid PR-AUC":round(average_precision_score(P.y[va],pv),4),"test PR-AUC":round(average_precision_score(T.y,s),4),"test ROC-AUC":round(roc_auc_score(T.y,s),4)}
    tt=T[["m","y"]].copy(); tt["s"]=s; tt["r"]=tt.groupby("m").s.rank(ascending=False,method="first")
    for K in [100,500]:
        hit=tt[tt.r<=K].groupby("m").y.sum(); o[f"P@{K}"]=round((hit/K).mean(),4); o[f"R@{K}"]=round(hit.sum()/tt.y.sum(),4)
    # bootstrap CI on test PR-AUC by month blocks
    ms=tt.m.unique(); g={m:i for m,i in tt.groupby("m").groups.items()}; b=[]
    for _ in range(100):
        pick=rs.choice(ms,len(ms)); ii=np.concatenate([g[m] for m in pick]); b.append(average_precision_score(tt.y.loc[ii],tt.s.loc[ii]))
    o["test PR-AUC 95% CI"]=f"{np.percentile(b,2.5):.4f}-{np.percentile(b,97.5):.4f}"
    res[name]=o
    if name=="+ keywords":
        coef=pd.Series(lr.coef_[0],index=cols); print("keyword coefs:",coef[kw].sort_values(ascending=False).round(3).to_dict())
print(pd.DataFrame(res).T.to_string())
json.dump(res,open(f"{d}/text_results.json","w"),ensure_ascii=False,indent=1)

"""개념 드리프트 실험: 학습 전략(고정 / 확장 / 최근 5년) x 모델(로지스틱 / LightGBM)을 연도별 테스트로 비교"""
import pandas as pd, numpy as np, sys, time, lightgbm as lgb, json
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d=sys.argv[1]; t0=time.time(); Y=lambda a,b:a*12+b-1; rs=np.random.default_rng(0)
P=pd.read_pickle(f"{d}/panel.pkl"); eps=0.5
P["growth3_12"]=(P.n3/3+eps)/(P.n12/12+eps); P["growth6_prev6"]=(P.n6+eps)/(P.n_prev6+eps); P["growth12_prev12"]=(P.n12+eps)/(P.n_prev12+eps)
for s in ["crash","fire","inj","dth"]: P[f"{s}_rate12"]=P[f"{s}12"]/P.n12
P["comp_c"]=P.comp.astype("category")
cnt=["n","n3","n6","n12","n24","n_prev6","n_prev12","n_cum","crash12","fire12","inj12","dth12","mon_since_first","age","prior_rcl","sib_n12","mc_n12","mmc_rcl12"]
grw=["growth3_12","growth6_prev6","growth12_prev12"]; rate=["crash_rate12","fire_rate12","inj_rate12","dth_rate12"]
feats=cnt+grw+rate+["comp_c"]
comps=sorted(P.comp.unique())
def XL(D):
    Z=pd.concat([np.log1p(D[cnt].clip(lower=0)),np.log(D[grw]),D[rate]],axis=1).astype(np.float32)
    oh=pd.get_dummies(pd.Categorical(D.comp,categories=comps),prefix="c",dtype=np.float32); oh.index=D.index
    return pd.concat([Z,oh],axis=1)
def sample(mask,nneg=700000):
    idx=P.index[mask]; y=P.y.values[mask]
    neg=idx[y==0]; neg=rs.choice(neg,min(nneg,len(neg)),replace=False)
    return np.concatenate([idx[y==1],neg]), len(neg)/max(1,(y==0).sum())
# train.py와 동일한 LightGBM 설정 + 동일한 early stopping 절차 (이전: lr 0.05 / min_data 1000 / 검증 없이 200트리 고정)
prm=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_data_in_leaf=2000,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,
         lambda_l2=10,cat_smooth=50,cat_l2=50,min_data_per_group=2000,verbose=-1,seed=0,num_threads=4)
VAL_M=12; GAP=6   # 각 학습 창의 마지막 12개월을 early-stopping 검증으로, 그 앞과는 6개월 엠바고
res=[]
for ty in [2022,2023,2024,2025]:
    te=(P.m>=Y(ty,1))&(P.m<=Y(ty,12)); T=P[te]
    cut=Y(ty,1)-7   # label horizon 6m: training rows must have fully observed labels before test start
    schemes={"fixed (2010-2019)":P.m<=Y(2019,12),"expanding (2010~y-1)":P.m<=cut,"sliding 5y":(P.m<=cut)&(P.m>cut-60)}
    for sn,mask in schemes.items():
        ii,_=sample(mask.values)
        S=P.loc[ii]
        Xs=XL(S); sc=StandardScaler().fit(Xs)
        lr=LogisticRegression(max_iter=300,C=0.5).fit(sc.transform(Xs),S.y)
        p_lr=lr.predict_proba(sc.transform(XL(T)))[:,1]
        mx=int(P.m[mask].max()); trm=mask&(P.m<=mx-VAL_M-GAP); vam=mask&(P.m>mx-VAL_M)
        jj,_=sample(trm.values); G=P.loc[jj]; V=P[vam]
        dtr=lgb.Dataset(G[feats],G.y); dva=lgb.Dataset(V[feats],V.y,reference=dtr)
        m=lgb.train(prm,dtr,1500,valid_sets=[dva],callbacks=[lgb.early_stopping(150,verbose=False)])
        p_gb=m.predict(T[feats],num_iteration=m.best_iteration)
        for mn,p in [("logistic",p_lr),("lightgbm",p_gb)]:
            tt=T[["m","y"]].copy(); tt["s"]=p; tt["r"]=tt.groupby("m").s.rank(ascending=False,method="first")
            hit=tt[tt.r<=100].groupby("m").y.sum()
            res.append(dict(test_year=ty,scheme=sn,model=mn,pr_auc=round(average_precision_score(tt.y,p),4),p_at_100=round((hit/100).mean(),4),base=round(tt.y.mean(),4)))
        print(ty,sn,f"{time.time()-t0:.0f}s",flush=True)
R=pd.DataFrame(res); R.to_csv(f"{d}/drift_results.csv",index=False)
print(R.pivot_table(index=["scheme","model"],columns="test_year",values="pr_auc").round(4).to_string())
print(R.pivot_table(index=["scheme","model"],columns="test_year",values="p_at_100").round(3).to_string())
print(R.groupby(["scheme","model"])[["pr_auc","p_at_100"]].mean().round(4).to_string())
# drift evidence: how positive rate by component shifts over time
P["yr"]=P.m//12
top=P.groupby("comp").y.sum().sort_values(ascending=False).index[:6]
print((P[P.comp.isin(top)].pivot_table(index="comp",columns="yr",values="y",aggfunc="mean")*100).round(2).loc[:,[2012,2015,2018,2021,2024]].to_string())

"""LightGBM 폭락 원인 분해(Ablation): 고정 학습(2010~2019, 6개월 엠바고)에서 피처 묶음을 하나씩 빼며 월 상위 100 정밀도(P@100)를 비교.
기존 주장 'LightGBM 패인 = 드리프트' 와 lgb_volume_only(8.4%) 결과의 충돌을 실험으로 설명하기 위한 스크립트.  사용: python src/ablation.py data [prr]"""
import pandas as pd, numpy as np, sys, lightgbm as lgb, json
from sklearn.metrics import average_precision_score
d=sys.argv[1]; USE_PRR=len(sys.argv)>2 and sys.argv[2]=="prr"; Y=lambda a,b:a*12+b-1
P=pd.read_pickle(f"{d}/panel_prr.pkl" if USE_PRR else f"{d}/panel.pkl"); eps=0.5
P["growth3_12"]=(P.n3/3+eps)/(P.n12/12+eps); P["growth6_prev6"]=(P.n6+eps)/(P.n_prev6+eps); P["growth12_prev12"]=(P.n12+eps)/(P.n_prev12+eps)
for s in ["crash","fire","inj","dth"]: P[f"{s}_rate12"]=P[f"{s}12"]/P.n12
P["comp_c"]=P.comp.astype("category")
tr=P.m<=Y(2019,6); va=(P.m>=Y(2020,1))&(P.m<=Y(2021,6)); te=P.m>=Y(2022,1)
full=["n","n3","n6","n12","n24","n_prev6","n_prev12","n_cum","crash12","fire12","inj12","dth12","crash_rate12","fire_rate12","inj_rate12","dth_rate12",
      "growth3_12","growth6_prev6","growth12_prev12","mon_since_first","age","prior_rcl","sib_n12","mc_n12","mmc_rcl12","comp_c"]
vol=["n","n3","n6","n12","n24","n_cum","mon_since_first","age","comp_c"]
V={"lgb_volume_only":vol,"lgb_full":full,
   "full - prior_rcl,mmc_rcl12 (반복 리콜 피처 제거)":[f for f in full if f not in ("prior_rcl","mmc_rcl12")],
   "full - sib_n12,mc_n12,mmc_rcl12 (타연식·동제조사 피처 제거)":[f for f in full if f not in ("sib_n12","mc_n12","mmc_rcl12")],
   "full - 심각도 8종":[f for f in full if "crash" not in f and "fire" not in f and "inj" not in f and "dth" not in f],
   "full - 증가율 3종":[f for f in full if not f.startswith("growth")],
   "full - comp_c (부품 범주 제거)":[f for f in full if f!="comp_c"],
   "volume_only + prior_rcl,mmc_rcl12":vol+["prior_rcl","mmc_rcl12"]}
if USE_PRR: V["full + PRR"]=full+["prr12"]; V["volume_only + PRR"]=vol+["prr12"]
prm=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_data_in_leaf=2000,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,
         lambda_l2=10,cat_smooth=50,cat_l2=50,min_data_per_group=2000,verbose=-1,seed=0,num_threads=4)   # train.py와 동일
T=P[te]; res={}
for name,feats in V.items():
    dtr=lgb.Dataset(P.loc[tr,feats],P.y[tr],categorical_feature=[c for c in feats if c=="comp_c"]); dva=lgb.Dataset(P.loc[va,feats],P.y[va],reference=dtr)
    m=lgb.train(prm,dtr,1500,valid_sets=[dva],callbacks=[lgb.early_stopping(150,verbose=False)]); s=m.predict(T[feats],num_iteration=m.best_iteration)
    tt=T[["m","y"]].copy(); tt["s"]=s; tt["r"]=tt.groupby("m").s.rank(ascending=False,method="first"); hit=tt[tt.r<=100].groupby("m").y.sum()
    res[name]=dict(best_iter=m.best_iteration,PR_AUC=round(average_precision_score(tt.y,s),4),P_at_100=round((hit/100).mean(),4)); print(name,res[name],flush=True)
R=pd.DataFrame(res).T; print("\n",R.to_string()); R.to_csv(f"{d}/ablation_results{'_prr' if USE_PRR else ''}.csv",encoding="utf-8-sig")

"""실제 차종-부품 사례 타임라인: 월별 불만 수 + 모델 위험 순위 + 미국 리콜 월 + 국내 리콜 월
사용: python src/case_study.py data MAKE MODEL YEAR COMP [KR_YYYYMM] [출력.png]   예) python src/case_study.py data BMW X5 2019 "ELECTRICAL SYSTEM" 202603
점수는 최근 5년 슬라이딩 재학습 로지스틱(triage.py와 동일), 순위는 같은 달 전체 키 중 순위."""
import pandas as pd, numpy as np, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d,mk,md,yr,comp=sys.argv[1:6]; kr=sys.argv[6] if len(sys.argv)>6 else None; out=sys.argv[7] if len(sys.argv)>7 else f"{d}/case_{mk}_{md}_{yr}.png"
mk=mk.upper(); md="".join(ch for ch in md.upper() if ch.isalnum()); comp=comp.upper(); Y=lambda a,b:a*12+b-1; rs=np.random.default_rng(0)
P=pd.read_pickle(f"{d}/panel.pkl"); K=pd.read_pickle(f"{d}/keys.pkl"); rk=pd.read_pickle(f"{d}/recall_units.pkl"); eps=0.5
P["growth3_12"]=(P.n3/3+eps)/(P.n12/12+eps); P["growth6_prev6"]=(P.n6+eps)/(P.n_prev6+eps); P["growth12_prev12"]=(P.n12+eps)/(P.n_prev12+eps)
for s in ["crash","fire","inj","dth"]: P[f"{s}_rate12"]=P[f"{s}12"]/P.n12
cnt=["n","n3","n6","n12","n24","n_prev6","n_prev12","n_cum","crash12","fire12","inj12","dth12","mon_since_first","age","prior_rcl","sib_n12","mc_n12","mmc_rcl12"]
grw=["growth3_12","growth6_prev6","growth12_prev12"]; rate=["crash_rate12","fire_rate12","inj_rate12","dth_rate12"]; comps=sorted(P.comp.unique())
def XL(D):
    Z=pd.concat([np.log1p(D[cnt].clip(lower=0)),np.log(D[grw]),D[rate]],axis=1).astype(np.float32)
    oh=pd.get_dummies(pd.Categorical(D.comp,categories=comps),prefix="c",dtype=np.float32); oh.index=D.index; return pd.concat([Z,oh],axis=1)
kid=K[(K.make==mk)&(K.model==md)&(K.year==yr)&(K.comp==comp)].index
if len(kid)==0: sys.exit(f"키 없음: {mk}|{md}|{yr}|{comp}")
kid=int(kid[0]); sc_rows=[]
for ty in [2022,2023,2024,2025,2026]:
    cut=Y(ty,1)-7; mask=((P.m<=cut)&(P.m>cut-60)).values; idx=P.index[mask]; y=P.y.values[mask]
    S=P.loc[np.concatenate([idx[y==1],rs.choice(idx[y==0],min(700000,int((y==0).sum())),replace=False)])]
    Xs=XL(S); sc=StandardScaler().fit(Xs); lr=LogisticRegression(max_iter=300,C=0.5).fit(sc.transform(Xs),S.y)
    T=P[(P.m>=Y(ty,1))&(P.m<=min(Y(ty,12),int(P.m.max())))]
    T=T.assign(s=lr.predict_proba(sc.transform(XL(T)))[:,1]); T["rank"]=T.groupby("m").s.rank(ascending=False,method="first"); sc_rows.append(T[T.kid==kid][["m","n","n3","n12","s","rank","y"]])
D=pd.concat(sc_rows)
if len(D)==0: sys.exit("이 키는 2022~ 점수 구간에 패널 행이 없음(최근 12개월 불만 0건이거나 리콜 직후 12개월 제외 구간)")
rm=sorted(rk[rk.kid==kid].m.unique())
fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True); x=D.m.values
ax[0].bar(x,D.n,color="#4c78a8"); ax[0].set_ylabel("monthly complaints")
ax[1].plot(x,D["rank"],color="#e45756",lw=2); ax[1].invert_yaxis(); ax[1].set_ylabel("risk rank (1=highest)"); ax[1].axhline(100,ls="--",c="gray",lw=.8); ax[1].axhline(500,ls=":",c="gray",lw=.8)
for r_ in rm:
    if x.min()<=r_<=x.max(): [a.axvline(r_,c="black",lw=2) for a in ax]; ax[0].text(r_,ax[0].get_ylim()[1]*.9," US recall",rotation=90,va="top")
if kr:
    k_=int(kr[:4])*12+int(kr[4:6])-1; [a.axvline(k_,c="green",lw=2,ls="--") for a in ax]; ax[0].text(k_,ax[0].get_ylim()[1]*.9," KR recall",rotation=90,va="top",color="green")
ticks=np.arange(x.min(),x.max()+1,6); ax[1].set_xticks(ticks); ax[1].set_xticklabels([f"{t//12}-{t%12+1:02d}" for t in ticks],rotation=45)
fig.suptitle(f"{mk} {md} {yr} / {comp}"); plt.tight_layout(); plt.savefig(out,dpi=150); print("saved",out)
print(D.assign(month=[f"{t//12}-{t%12+1:02d}" for t in D.m]).drop(columns="m").round(4).to_string(index=False))

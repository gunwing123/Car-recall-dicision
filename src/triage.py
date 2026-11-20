"""오경보 해부 + 2단계 선별(Triage) 큐 + 최신 워치리스트
사용: python src/triage.py data [data/kr_recall_auto.csv]
1) 최근 5년 슬라이딩 재학습 로지스틱(드리프트 실험 최선 전략)으로 2022.01~2026.03 매월 점수 산출 (6개월 엠바고)
2) 월 상위 K 경보를 정체별로 분류: 적중(6개월 내 리콜) / 지연 적중(7~12개월) / 연식 확장(같은 차종-부품 타연식 12개월 내 리콜) / 고판매량 의심 / 기타
3) 1단계(상위 500) -> 2단계 필터(심각도>=1건 또는 급증) -> 월 N건 점검 큐의 정밀도·포착량 비교, 확률 보정(언더샘플링 보정)
4) 가장 최근 월 기준 수입차 워치리스트(국내 리콜 미발표 키) 출력 (kr csv를 주면)"""
import pandas as pd, numpy as np, sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d=sys.argv[1]; kr_path=sys.argv[2] if len(sys.argv)>2 else None
Y=lambda a,b:a*12+b-1; rs=np.random.default_rng(0); NNEG=700000
P=pd.read_pickle(f"{d}/panel.pkl"); K=pd.read_pickle(f"{d}/keys.pkl"); rk=pd.read_pickle(f"{d}/recall_units.pkl")
P=P.join(K[["model"]],on="kid")   ; eps=0.5   # P에는 mmc(같은 제조사-차종-부품 그룹)가 이미 있음; eps=0.5
P["growth3_12"]=(P.n3/3+eps)/(P.n12/12+eps); P["growth6_prev6"]=(P.n6+eps)/(P.n_prev6+eps); P["growth12_prev12"]=(P.n12+eps)/(P.n_prev12+eps)
for s in ["crash","fire","inj","dth"]: P[f"{s}_rate12"]=P[f"{s}12"]/P.n12
P["sev12"]=P.crash12+P.fire12+P.inj12+P.dth12
cnt=["n","n3","n6","n12","n24","n_prev6","n_prev12","n_cum","crash12","fire12","inj12","dth12","mon_since_first","age","prior_rcl","sib_n12","mc_n12","mmc_rcl12"]
grw=["growth3_12","growth6_prev6","growth12_prev12"]; rate=["crash_rate12","fire_rate12","inj_rate12","dth_rate12"]; comps=sorted(P.comp.unique())
def XL(D):
    Z=pd.concat([np.log1p(D[cnt].clip(lower=0)),np.log(D[grw]),D[rate]],axis=1).astype(np.float32)
    oh=pd.get_dummies(pd.Categorical(D.comp,categories=comps),prefix="c",dtype=np.float32); oh.index=D.index; return pd.concat([Z,oh],axis=1)
parts=[]; TEST_END=int(P.m.max())
for ty in [2022,2023,2024,2025,2026]:
    cut=Y(ty,1)-7; mask=((P.m<=cut)&(P.m>cut-60)).values; idx=P.index[mask]; y=P.y.values[mask]
    neg=idx[y==0]; nk=min(NNEG,len(neg)); beta=nk/len(neg)      # beta: 음성 유지 비율 (확률 보정용)
    S=P.loc[np.concatenate([idx[y==1],rs.choice(neg,nk,replace=False)])]
    Xs=XL(S); sc=StandardScaler().fit(Xs); lr=LogisticRegression(max_iter=300,C=0.5).fit(sc.transform(Xs),S.y)
    T=P[(P.m>=Y(ty,1))&(P.m<=min(Y(ty,12),TEST_END))].copy()
    p=lr.predict_proba(sc.transform(XL(T)))[:,1]; T["s"]=p; T["s_cal"]=p*beta/(p*beta+(1-p))   # prior-odds 보정: 언더샘플링 전 base rate로 환산
    parts.append(T); print(ty,"scored",len(T),flush=True)
T=pd.concat(parts); T["rank"]=T.groupby("m").s.rank(ascending=False,method="first")
rmax=int(rk.m.max())
# 같은 차종-부품(mmc)의 '다른 연식' 12개월 내 리콜 / 같은 키 7~12개월 지연 리콜
rk2=rk.join(K[["mmc"]],on="kid")
by_kid={k_:np.sort(v.values) for k_,v in rk.groupby("kid").m}; by_mmc={k_:np.sort(v.values) for k_,v in rk2.groupby("mmc").m}
def has(arr_map,key,lo,hi):
    a=arr_map.get(key); 
    return False if a is None else (np.searchsorted(a,hi,side="right")>np.searchsorted(a,lo,side="left"))
def classify(df):
    out=[]
    for kid,mmc,m,y,n12 in zip(df.kid,df.mmc,df.m,df.y,df.n12):
        if y==1: out.append("적중(1~6개월 내 리콜)")
        elif has(by_kid,kid,m+7,m+12): out.append("지연 적중(7~12개월 내 리콜)")
        elif has(by_mmc,mmc,m+1,m+12): out.append("연식 확장(같은 차종-부품 타연식 12개월 내 리콜)")
        else: out.append("기타(리콜 없음)")
    return np.array(out)
obs=T[T.m<=rmax-12].copy()      # 12개월 미래가 관측된 달만
res=[]
for K_ in [100,500]:
    F=obs[obs["rank"]<=K_].copy(); F["type"]=classify(F)
    q90=obs.groupby("m").n12.transform(lambda s:s.quantile(0.9)).loc[F.index]
    F.loc[(F.type=="기타(리콜 없음)")&(F.n12>=q90),"type"]="기타 중 고불만량(상위 10%) - 판매량 효과 의심"
    sh=F.type.value_counts(normalize=True).round(3); print(f"\n[월 상위 {K_}] 경보 {len(F)}건(12개월 관측 달만) 구성"); print(sh.to_string())
    res.append(sh.rename(f"top{K_}"))
pd.concat(res,axis=1).fillna(0).to_csv(f"{d}/false_alarm_breakdown.csv",encoding="utf-8-sig")
# ---- 2단계 Triage
rows=[]
for M_ in [20,30,50]:
    for name in ["단순 상위 N","2단계(상위500 -> 심각도>=1 또는 급증 상위30% -> 점수순 N)"]:
        hits=[];prec=[]
        for m,g in obs.groupby("m"):
            if name.startswith("단순"): q=g.nsmallest(M_,"rank")
            else:
                s1=g[g["rank"]<=500]; thr=s1.growth3_12.quantile(0.7)
                q=s1[(s1.sev12>=1)|(s1.growth3_12>=thr)].nlargest(M_,"s")
            hits.append(q.y.sum()); prec.append(q.y.mean() if len(q) else np.nan)
        tot=obs.groupby("m").y.sum()
        rows.append(dict(월점검건수=M_,방식=name,정밀도=round(np.nanmean(prec),4),월평균_적중=round(np.mean(hits),2),전체양성대비_포착률=round(np.sum(hits)/tot.sum(),4)))
TR=pd.DataFrame(rows); print("\n[점검 큐 비교]\n",TR.to_string(index=False)); TR.to_csv(f"{d}/triage_queue.csv",index=False,encoding="utf-8-sig")
print("\n[확률 보정 후] 상위 100 평균 보정확률 %.3f vs 실제 적중률 %.3f"%(obs[obs['rank']<=100].s_cal.mean(),obs[obs['rank']<=100].y.mean()))
# ---- 워치리스트
if kr_path:
    from kr_common import bm, load_kr, cat, DOMESTIC, model_tokens
    last=int(T.m.max()); L=T[(T.m==last)&(T["rank"]<=500)].copy()
    imp={mk for b,mks in bm.items() if b not in DOMESTIC for mk in mks}; L=L[L.make.isin(imp)]
    k=load_kr(kr_path); k=k[~k.emission&k.brand.isin(bm)&~k.brand.isin(DOMESTIC)]
    def done(x):
        for kk in k.itertuples(index=False):
            if x.make in bm.get(kk.brand,[]) and (kk.comp==x.comp or kk.comp=="OTHER/STANDARD") and any(t in x.model or (len(x.model)>=2 and x.model in t) for t in model_tokens(kk.modlNmInfo)): return True
        return False
    L["kr_recall_exists"]=[done(x) for x in L.itertuples(index=False)]
    W=L[~L.kr_recall_exists].nlargest(10,"s")[["make","model","comp","n12","n3","sev12","growth3_12","s","s_cal","rank"]]
    W.insert(0,"asof",f"{last//12}-{last%12+1:02d}"); print("\n[수입차 선제 점검 워치리스트 Top 10 - 국내 리콜 미발표(2025+ 소비자24 기준)]\n",W.to_string(index=False)); W.to_csv(f"{d}/watchlist_top10.csv",index=False,encoding="utf-8-sig")

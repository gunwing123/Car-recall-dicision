"""새 가설 H5: 미국 NHTSA 조기경보 모델이 국내 수입차 리콜을 미리 잡는가?
- 연도별로 최근 5년 데이터로 재학습한 로지스틱(드리프트 실험의 최선 전략)으로 2024.01~2026.03 매달 미국 차종-연식-부품 위험 점수 산출
- 국내 리콜(소비자24)마다 같은 브랜드·차종명·부품의 미국 키가 국내 공표 전 12개월 안에 월 상위 K에 든 적이 있는지 확인
- 비교 기준: 같은 브랜드의 국내 리콜이 없는 미국 차종-부품 그룹이 같은 창에서 상위 K에 든 비율"""
import pandas as pd, numpy as np, re, sys, importlib.util
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d, kr_path = sys.argv[1], sys.argv[2]; Y=lambda a,b:a*12+b-1; rs=np.random.default_rng(0)
spec=importlib.util.spec_from_file_location("kv",__file__.replace("us_model_to_kr.py","kr_vs_us.py"))
src=open(spec.origin).read(); bm=eval(src[src.index("bm={")+3:src.index("}",src.index("\"한국모터트레이딩\""))+1])
rules=eval(src[src.index("rules=[")+6:src.index("]\ndef cat")+1])
P=pd.read_pickle(f"{d}/panel.pkl"); K=pd.read_pickle(f"{d}/keys.pkl")
P=P.join(K[["model"]],on="kid"); eps=0.5
P["growth3_12"]=(P.n3/3+eps)/(P.n12/12+eps); P["growth6_prev6"]=(P.n6+eps)/(P.n_prev6+eps); P["growth12_prev12"]=(P.n12+eps)/(P.n_prev12+eps)
for s in ["crash","fire","inj","dth"]: P[f"{s}_rate12"]=P[f"{s}12"]/P.n12
cnt=["n","n3","n6","n12","n24","n_prev6","n_prev12","n_cum","crash12","fire12","inj12","dth12","mon_since_first","age","prior_rcl","sib_n12","mc_n12","mmc_rcl12"]
grw=["growth3_12","growth6_prev6","growth12_prev12"]; rate=["crash_rate12","fire_rate12","inj_rate12","dth_rate12"]; comps=sorted(P.comp.unique())
def XL(D):
    Z=pd.concat([np.log1p(D[cnt].clip(lower=0)),np.log(D[grw]),D[rate]],axis=1).astype(np.float32)
    oh=pd.get_dummies(pd.Categorical(D.comp,categories=comps),prefix="c",dtype=np.float32); oh.index=D.index; return pd.concat([Z,oh],axis=1)
scores=[]
for ty in [2024,2025,2026]:
    cut=Y(ty,1)-7; mask=((P.m<=cut)&(P.m>cut-60)).values; idx=P.index[mask]; y=P.y.values[mask]
    ii=np.concatenate([idx[y==1],rs.choice(idx[y==0],700000,replace=False)]); S=P.loc[ii]
    Xs=XL(S); sc=StandardScaler().fit(Xs); lr=LogisticRegression(max_iter=300,C=0.5).fit(sc.transform(Xs),S.y)
    T=P[(P.m>=Y(ty,1))&(P.m<=min(Y(ty,12),Y(2026,3)))]
    scores.append(T[["kid","m","make","model","comp"]].assign(s=lr.predict_proba(sc.transform(XL(T)))[:,1]))
Sc=pd.concat(scores); Sc["rank"]=Sc.groupby("m").s.rank(ascending=False,method="first")
print("scored months",Sc.m.min()//12,Sc.m.min()%12+1,"~",Sc.m.max()//12,Sc.m.max()%12+1,"| keys/month",int(Sc.groupby("m").size().mean()))
# Korean recalls (imported brands, 국토부)
k=pd.read_csv(kr_path,dtype=str,encoding="utf-8-sig"); k["date"]=pd.to_datetime(k.recallPublictBgnde,format="%Y%m%d",errors="coerce")
k=k[(k.date>="2025-01-01")&~k.infoOriginInstt.str.contains("환경부",na=False)].copy()
k["brand"]=k.productNm.str.extract(r"^\[(.+?)\]")[0].str.strip()
txt=k.productNm.str.split(" - ",n=1).str[1].fillna("")+" "+k.shrtcomCn.fillna("").str.replace(r"<[^>]+>"," ",regex=True)
def cat(s):
    for c,p in rules:
        if re.search(p,s): return c
    return None
k["comp"]=[cat(s) for s in txt]
dom={"현대","기아","제네시스","케이지모빌리티","르노","르노삼성"}
k=k[k.brand.isin(bm)&~k.brand.isin(dom)&k.comp.notna()]
# map KR coarse comp names to NHTSA top-level names used in panel
alias={"FUEL SYSTEM":["FUEL SYSTEM, GASOLINE","FUEL SYSTEM, DIESEL","FUEL SYSTEM, OTHER"],"SERVICE BRAKES":["SERVICE BRAKES, HYDRAULIC","SERVICE BRAKES, ELECTRIC","SERVICE BRAKES, AIR","SERVICE BRAKES"],
 "TIRES/WHEELS":["TIRES","WHEELS"],"ENGINE":["ENGINE","ENGINE AND ENGINE COOLING"],"LATCHES/LOCKS":["LATCHES/LOCKS/LINKAGES"],
 "ELECTRICAL SYSTEM":["ELECTRICAL SYSTEM","FORWARD COLLISION AVOIDANCE","LANE DEPARTURE","HYBRID PROPULSION SYSTEM","COMMUNICATION"]}
norm=lambda s:re.sub(r"[^A-Z0-9]","",str(s).upper())
for K_ in [100,500]:
    flag=Sc[Sc["rank"]<=K_]
    out=[]
    for _,x in k.iterrows():
        makes=bm[x.brand]; cs=alias.get(x.comp,[x.comp]); kd=x.date.year*12+x.date.month-1
        toks={norm(t.split()[0]) for t in re.split(r"[,/]",str(x.modlNmInfo)) if t.strip()}; toks={t for t in toks if len(t)>=2}
        g=Sc[Sc.make.isin(makes)&Sc.comp.isin(cs)&(Sc.m>=kd-12)&(Sc.m<kd)]
        g=g[[any(t in m or (len(m)>=2 and m in t) for t in toks) for m in g.model]]
        if len(g)==0: out.append((x.recallSn,"US key not in panel",np.nan)); continue
        f=g[g["rank"]<=K_]
        out.append((x.recallSn,"flagged" if len(f) else "not flagged",kd-f.m.min() if len(f) else np.nan))
    O=pd.DataFrame(out,columns=["recallSn","status","lead_months"])
    ev=O[O.status!="US key not in panel"]
    # baseline: model-comp groups of the same imported makes, any month in a random 12m window
    allg=Sc[Sc.make.isin(sum([bm[b] for b in k.brand.unique()],[]))].copy(); allg["grp"]=allg.make+"|"+allg.model+"|"+allg.comp
    base=[]
    for kd in (k.date.dt.year*12+k.date.dt.month-1).sample(50,random_state=0,replace=True):
        w=allg[(allg.m>=kd-12)&(allg.m<kd)]; base.append(w.groupby("grp")["rank"].min().le(K_).mean())
    print(f"\n[월 상위 {K_}] 국내 수입차 리콜 {len(O)}건 중 미국 키 존재 {len(ev)}건 → 사전 포착 {int((ev.status=='flagged').sum())}건 "
          f"({(ev.status=='flagged').mean():.1%}), 리드타임 중간값 {ev.lead_months.median():.0f}개월 | 비교 기준(임의 차종-부품 그룹) {np.mean(base):.1%}")
    O.to_csv(f"{d}/us_model_to_kr_top{K_}.csv",index=False)

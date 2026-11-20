"""가설 H5: 미국 NHTSA 조기경보 모델이 국내 수입차 리콜을 미리 잡는가?
- 연도별로 최근 5년 데이터로 재학습한 로지스틱(드리프트 실험의 최선 전략)으로 2024.01~2026.03 매달 미국 차종-연식-부품 위험 점수 산출
- 국내 리콜(소비자24)마다 같은 브랜드·차종명·부품의 미국 키가 국내 공표 전 12개월 안에 월 상위 K에 든 적이 있는지 확인

[수정 사항 - 비교 기준(베이스라인) 불일치 해소]
 기존: 평가 대상은 '여러 차종 토큰 x 모든 연식 x 부품 별칭 합집합' 중 하나라도 상위 K면 포착, 비교 기준은 '단일 차종-단일 부품'의 평균 -> 집계 단위가 달라 5배가 부풀 수 있음.
 수정: (1) 같은 국내 리콜의 같은 키 그룹(동일 매칭 규칙)을 '다른 달'에 평가하는 placebo를 비교 기준으로 사용 (집계 단위 동일),
       (2) 경보 직전 12개월 안에 같은 그룹의 미국 리콜이 이미 있었던 달은 '2단계(리콜 후)'로 분리하고 '1단계(순수 사전 경보)'만 따로 집계,
       (3) 전체 국내 수입차 리콜(분모)과 미국 키가 존재하는 리콜을 모두 보고, 리드타임이 12개월에 몰리는지 분포를 출력."""
import pandas as pd, numpy as np, re, sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from kr_common import bm, rules, cat, norm, model_tokens, load_kr, DOMESTIC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d, kr_path = sys.argv[1], sys.argv[2]; Y=lambda a,b:a*12+b-1; rs=np.random.default_rng(0)
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
# ---- Korean recalls (imported brands, 국토부)
k=load_kr(kr_path); k=k[~k.emission]
k["comp"]=[cat(t) if cat(t)!="OTHER/STANDARD" else None for t in k.txt]
n_all_import=int((k.brand.isin(bm)&~k.brand.isin(DOMESTIC)).sum())
k=k[k.brand.isin(bm)&~k.brand.isin(DOMESTIC)&k.comp.notna()].reset_index(drop=True)
print(f"국내 수입차 리콜(국토부) {n_all_import}건 중 부품 분류 가능 {len(k)}건")
alias={"FUEL SYSTEM":["FUEL SYSTEM, GASOLINE","FUEL SYSTEM, DIESEL","FUEL SYSTEM, OTHER"],"SERVICE BRAKES":["SERVICE BRAKES, HYDRAULIC","SERVICE BRAKES, ELECTRIC","SERVICE BRAKES, AIR","SERVICE BRAKES"],
 "TIRES/WHEELS":["TIRES","WHEELS"],"ENGINE":["ENGINE","ENGINE AND ENGINE COOLING"],"LATCHES/LOCKS":["LATCHES/LOCKS/LINKAGES"],
 "ELECTRICAL SYSTEM":["ELECTRICAL SYSTEM","FORWARD COLLISION AVOIDANCE","LANE DEPARTURE","HYBRID PROPULSION SYSTEM","COMMUNICATION"]}
rk=pd.read_pickle(f"{d}/recall_units.pkl")            # kid, m, campno (미국 리콜 접수 월)
rcl_by_kid=rk.groupby("kid").m.apply(lambda s:np.sort(s.values)).to_dict()
Smin,Smax=int(Sc.m.min()),int(Sc.m.max())

def group_rows(x):
    """국내 리콜 x와 같은 그룹(브랜드 -> 미국 제조사, 부품 별칭, 차종 토큰)의 미국 키-월 점수. 평가 대상과 placebo가 같은 집계 단위를 쓰도록 한 번만 계산."""
    makes=bm[x.brand]; cs=alias.get(x.comp,[x.comp]); toks=model_tokens(x.modlNmInfo)
    g=Sc[Sc.make.isin(makes)&Sc.comp.isin(cs)]
    ok={m for m in g.model.unique() if any(t in m or (len(m)>=2 and m in t) for t in toks)}
    g=g[g.model.isin(ok)].copy()
    if len(g):
        RM=np.sort(np.concatenate([rcl_by_kid.get(i,np.array([],int)) for i in g.kid.unique()]))
        g["post_us"]=np.searchsorted(RM,g.m.values,side="right")>np.searchsorted(RM,g.m.values-12,side="left")  # 직전 12개월 안에 같은 그룹 미국 리콜 있음 = 2단계
    else: g["post_us"]=False
    return g
def flag(g,lo,hi,K_,stage1_only):
    w=g[(g.m>=lo)&(g.m<hi)&(g["rank"]<=K_)]
    if stage1_only: w=w[~w.post_us]
    return w
groups=[group_rows(x) for x in k.itertuples(index=False)]
kds=(k.date.dt.year*12+k.date.dt.month-1).values
valid_kd=np.arange(Smin+12,Smax+2)                    # 12개월 창이 점수 구간 안에 완전히 들어오는 공표월 후보
for K_ in [100,500]:
    rows=[]
    for x,g,kd in zip(k.itertuples(index=False),groups,kds):
        full=(kd-12>=Smin)&(kd-1<=Smax)
        if len(g[(g.m>=kd-12)&(g.m<kd)])==0: rows.append((x.recallSn,x.brand,x.comp,kd,"US key not in panel",np.nan,np.nan,np.nan,np.nan,full)); continue
        f=flag(g,kd-12,kd,K_,False); f1=flag(g,kd-12,kd,K_,True)
        # placebo: 같은 그룹을 다른 공표월(실제와 6개월 이상 떨어진 달, 없으면 전체)에 평가 -> 차종-부품의 '상시 상위권 체류' 효과를 그대로 반영
        alt=[v for v in valid_kd if abs(v-kd)>=6] or list(valid_kd)
        pl=np.mean([len(flag(g,v-12,v,K_,False))>0 for v in alt]); pl1=np.mean([len(flag(g,v-12,v,K_,True))>0 for v in alt])
        rows.append((x.recallSn,x.brand,x.comp,kd,"flagged" if len(f) else "not flagged",kd-f.m.min() if len(f) else np.nan,float(len(f1)>0),pl,pl1,full))
    O=pd.DataFrame(rows,columns=["recallSn","brand","comp","kd","status","lead_months","flag_stage1","placebo_any","placebo_stage1","full_window"])
    O.to_csv(f"{d}/us_model_to_kr_top{K_}.csv",index=False)
    ev=O[O.status!="US key not in panel"]; fl=ev[ev.status=="flagged"]
    def line(E,tag):
        if len(E)==0: return
        fa=(E.status=="flagged").mean(); f1=E.flag_stage1.mean(); pa=E.placebo_any.mean(); p1=E.placebo_stage1.mean()
        print(f"  {tag:<28} n={len(E):>3} | 포착(전체) {fa:6.1%} vs placebo {pa:6.1%} (x{fa/max(pa,1e-9):.1f}) | 포착(1단계만) {f1:6.1%} vs placebo {p1:6.1%} (x{f1/max(p1,1e-9):.1f})")
    print(f"\n[월 상위 {K_}] 국내 수입차 리콜 {len(O)}건 중 미국 키 존재 {len(ev)}건({len(ev)/len(O):.1%}), 사전 포착 {len(fl)}건")
    print(f"  분모를 전체 {len(O)}건으로 보면 포착률 {len(fl)/len(O):.1%} (미국 키 존재 {len(ev)}건 기준 {len(fl)/max(len(ev),1):.1%})")
    line(ev,"전체"); line(ev[ev.full_window],"12개월 창 완전 관측만")
    if len(fl):
        ld=fl.lead_months
        print(f"  리드타임 분포: 중간값 {ld.median():.0f}개월 | 11~12개월 {(ld>=11).mean():.0%} | 12개월 정확히 {(ld==12).mean():.0%} -> 높을수록 '상시 상위권 체류' 의심")
        print(f"  1단계(미국 리콜 전 순수 경보) {int((fl.flag_stage1==1).sum())}건 / 포착 {len(fl)}건")

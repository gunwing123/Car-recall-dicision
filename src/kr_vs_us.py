"""국내(소비자24) vs 미국(NHTSA) 자동차 리콜 비교: 부품 분포 + 같은 차종·부품 리콜의 시차"""
import pandas as pd, numpy as np, re, sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from kr_common import *
kr_path, d = sys.argv[1], sys.argv[2]
k=pd.read_csv(kr_path,dtype=str,encoding="utf-8-sig")
k["date"]=pd.to_datetime(k.recallPublictBgnde,format="%Y%m%d",errors="coerce")
k=k[k.date>="2025-01-01"].copy()  # API가 사실상 최근 리콜만 제공
k["brand"]=k.productNm.str.extract(r"^\[(.+?)\]")[0].str.strip()
k["defect"]=k.productNm.str.split(" - ",n=1).str[1].fillna("")
txt=(k.defect+" "+k.shrtcomCn.fillna("").str.replace(r"<[^>]+>"," ",regex=True))
k["comp"]=[cat(s) for s in txt]
k["emission"]=k.infoOriginInstt.str.contains("환경부",na=False)
print("국내 리콜",len(k),"| 기간",k.date.min().date(),"~",k.date.max().date(),"| 환경부(배출가스)",int(k.emission.sum()),"| 국토부",int((~k.emission).sum()))
ks=k[~k.emission]
r=load_us_recalls(d)
us=r[(r.date>=k.date.min())&(r.date<=k.date.max())].drop_duplicates(["campno","comp1"])
dist=pd.DataFrame({"KR_share":ks.comp.value_counts(normalize=True),"US_share":us.comp1.value_counts(normalize=True)}).fillna(0)
dist=dist[(dist.KR_share>0)|(dist.US_share>=0.02)].sort_values("KR_share",ascending=False).round(3)
print("\n부품 분포 (같은 기간, 국토부 리콜 vs NHTSA 캠페인)\n",dist.to_string())
# 같은 브랜드 + 차종 토큰 + 부품 분류의 미국 리콜 vs 국내 리콜 날짜 (공통 매칭 함수 사용)
US_END=r.date.max()
def run(past_m, fut_m, censor_safe=False, quiet=False):
    """past_m/fut_m: 국내 공표일 기준 미국 리콜을 찾는 과거/미래 창(개월). censor_safe=True면 미래 창이 미국 데이터 끝(US_END)을 넘지 않는 국내 리콜만 사용."""
    rows=[]
    for _,x in ks.iterrows():
        if censor_safe and x.date+pd.Timedelta(days=round(fut_m*30.44))>US_END: continue
        c=match_us(x,r,round(past_m*30.44),round(fut_m*30.44))
        if c is None: rows.append((x.recallSn,x.brand,"no-brand-map",np.nan,None)); continue
        if len(c)==0: rows.append((x.recallSn,x.brand,"no-match",np.nan,None)); continue
        lag=(x.date-c.date).dt.days/30.44; i=lag.abs().idxmin()
        rows.append((x.recallSn,x.brand,"match",round(lag[i],1),c.loc[i,"campno"]))
    return pd.DataFrame(rows,columns=["recallSn","brand","status","lag_months_kr_minus_us","us_campno"])
def summarize(M):
    m=M[M.status=="match"].lag_months_kr_minus_us
    return dict(n_kr=len(M),n_match=len(m),us_first=int((m>0.5).sum()),similar=int((m.abs()<=0.5).sum()),kr_first=int((m<-0.5).sum()),
                us_first_share=round((m>0.5).mean(),3) if len(m) else np.nan,median_lag=round(m.median(),1) if len(m) else np.nan)
# (1) 원래 설정: 과거 36개월 / 미래 12개월 (비대칭 -> 미국 선행 비율이 부풀 수 있음)
M=run(36,12)
print("\n매칭 상태 [원래 비대칭 창 -36/+12개월]\n",M.status.value_counts().to_string())
S0=summarize(M)
print(f"\n[원래 창] 매칭 {S0['n_match']}건: 미국이 먼저 {S0['us_first']}건, 비슷(±0.5개월) {S0['similar']}건, 한국이 먼저 {S0['kr_first']}건, 시차 중간값 {S0['median_lag']}개월")
print("국내 수입 브랜드 vs 국산 매칭률:",M.assign(dom=M.brand.isin(DOMESTIC)).groupby("dom").status.apply(lambda s:round((s=="match").mean(),2)).to_dict())
print("브랜드별 매칭률\n",M.groupby("brand").status.apply(lambda s:(s=="match").mean()).sort_values(ascending=False).round(2).head(15).to_string())
k[["recallSn","brand","comp","date","emission"]].merge(M,on=["recallSn","brand"],how="left").to_csv(f"{d}/kr_vs_us_matches.csv",index=False,encoding="utf-8-sig")
dist.to_csv(f"{d}/kr_vs_us_component_share.csv",encoding="utf-8-sig")
# (2) 대칭 창 민감도: 과거/미래 창을 같게 두고, 미래 창이 미국 데이터 끝을 넘는 국내 리콜(우측 중도절단)은 제외
sens=[]
for name,(p,f,cs) in {"원래(-36/+12, 절단 포함)":(36,12,False),"원래 창 + 절단 제외":(36,12,True),"대칭 ±12 + 절단 제외":(12,12,True),
                      "대칭 ±6 + 절단 제외":(6,6,True),"대칭 ±24 + 절단 제외":(24,24,True)}.items():
    S=summarize(run(p,f,cs)); S["window"]=name; sens.append(S)
SD=pd.DataFrame(sens).set_index("window")
print("\n[H4 민감도] 미국이 먼저 나온 비율은 창이 대칭일 때도 유지되는가?\n",SD.to_string())
SD.to_csv(f"{d}/kr_vs_us_window_sensitivity.csv",encoding="utf-8-sig")

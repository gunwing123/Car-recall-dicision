"""국내(소비자24) vs 미국(NHTSA) 자동차 리콜 비교: 부품 분포 + 같은 차종·부품 리콜의 시차"""
import pandas as pd, numpy as np, re, sys
kr_path, d = sys.argv[1], sys.argv[2]
k=pd.read_csv(kr_path,dtype=str,encoding="utf-8-sig")
k["date"]=pd.to_datetime(k.recallPublictBgnde,format="%Y%m%d",errors="coerce")
k=k[k.date>="2025-01-01"].copy()  # API가 사실상 최근 리콜만 제공
k["brand"]=k.productNm.str.extract(r"^\[(.+?)\]")[0].str.strip()
k["defect"]=k.productNm.str.split(" - ",n=1).str[1].fillna("")
txt=(k.defect+" "+k.shrtcomCn.fillna("").str.replace(r"<[^>]+>"," ",regex=True))
rules=[("AIR BAGS",r"에어백"),("SEAT BELTS",r"안전띠|안전벨트|좌석벨트|시트벨트"),("BACK OVER PREVENTION",r"후방.{0,4}카메라|후방 영상|후방카메라"),
 ("SERVICE BRAKES",r"브레이크|제동"),("STEERING",r"조향|스티어링"),("FUEL SYSTEM",r"연료"),("POWER TRAIN",r"변속|트랜스미션|구동축|드라이브샤프트|파워트레인"),
 ("EXTERIOR LIGHTING",r"등화|램프|전조등|헤드라이트|라이트|방향지시"),("TIRES/WHEELS",r"타이어|휠"),("SUSPENSION",r"서스펜션|현가"),
 ("SEATS",r"좌석|시트"),("LATCHES/LOCKS",r"도어|잠금|래치|후드"),("VISIBILITY",r"와이퍼|유리|미러|거울"),
 ("ENGINE",r"엔진|크랭크|냉각"),("ELECTRICAL SYSTEM",r"배터리|고전압|전기|전원|소프트웨어|계기판|디스플레이|배선|퓨즈|컨트롤 ?유닛|ECU")]
def cat(s):
    for c,p in rules:
        if re.search(p,s): return c
    return "OTHER/STANDARD"
k["comp"]=[cat(s) for s in txt]
k["emission"]=k.infoOriginInstt.str.contains("환경부",na=False)
print("국내 리콜",len(k),"| 기간",k.date.min().date(),"~",k.date.max().date(),"| 환경부(배출가스)",int(k.emission.sum()),"| 국토부",int((~k.emission).sum()))
ks=k[~k.emission]
r=pd.read_csv(f"{d}/FLAT_RCL_POST_2010.txt",sep="\t",header=None,usecols=[1,2,3,6,10,15],names=["campno","make","model","comp","type","rcdate"],
   dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
r=r[r.type=="V"]; r["date"]=pd.to_datetime(r.rcdate,format="%Y%m%d",errors="coerce")
r["comp1"]=r.comp.str.split(":").str[0].str.strip().str.upper()
r["comp1"]=r.comp1.replace({"FUEL SYSTEM, GASOLINE":"FUEL SYSTEM","FUEL SYSTEM, DIESEL":"FUEL SYSTEM","FUEL SYSTEM, OTHER":"FUEL SYSTEM",
   "SERVICE BRAKES, HYDRAULIC":"SERVICE BRAKES","SERVICE BRAKES, AIR":"SERVICE BRAKES","SERVICE BRAKES, ELECTRIC":"SERVICE BRAKES",
   "WHEELS":"TIRES/WHEELS","TIRES":"TIRES/WHEELS","ENGINE AND ENGINE COOLING":"ENGINE","LATCHES/LOCKS/LINKAGES":"LATCHES/LOCKS","FORWARD COLLISION AVOIDANCE":"ELECTRICAL SYSTEM",
   "LANE DEPARTURE":"ELECTRICAL SYSTEM","ELECTRONIC STABILITY CONTROL (ESC)":"SERVICE BRAKES","HYBRID PROPULSION SYSTEM":"ELECTRICAL SYSTEM","COMMUNICATION":"ELECTRICAL SYSTEM"})
us=r[(r.date>=k.date.min())&(r.date<=k.date.max())].drop_duplicates(["campno","comp1"])
dist=pd.DataFrame({"KR_share":ks.comp.value_counts(normalize=True),"US_share":us.comp1.value_counts(normalize=True)}).fillna(0)
dist=dist[(dist.KR_share>0)|(dist.US_share>=0.02)].sort_values("KR_share",ascending=False).round(3)
print("\n부품 분포 (같은 기간, 국토부 리콜 vs NHTSA 캠페인)\n",dist.to_string())
# same brand + model token + component: US recall date vs KR date
bm={"벤츠":["MERCEDES-BENZ","MERCEDES BENZ"],"BMW":["BMW"],"미니":["MINI"],"혼다":["HONDA","ACURA"],"토요타":["TOYOTA"],"렉서스":["LEXUS"],
    "포드":["FORD"],"링컨":["LINCOLN"],"에프엘오토":["FORD","LINCOLN"],"폭스바겐":["VOLKSWAGEN"],"아우디":["AUDI"],"볼보":["VOLVO"],"지프":["JEEP"],
    "스텔란티스":["JEEP","CHRYSLER","DODGE","RAM","PEUGEOT"],"포르쉐":["PORSCHE"],"재규어랜드로버":["LAND ROVER","JAGUAR"],"랜드로버":["LAND ROVER"],
    "현대":["HYUNDAI","GENESIS"],"기아":["KIA"],"제네시스":["GENESIS"],"한국지엠":["CHEVROLET","GMC","BUICK","CADILLAC"],"쉐보레":["CHEVROLET"],
    "캐딜락":["CADILLAC"],"테슬라":["TESLA"],"닛산":["NISSAN"],"인피니티":["INFINITI"],"스바루":["SUBARU"],"마세라티":["MASERATI"],"페라리":["FERRARI"],
    "람보르기니":["LAMBORGHINI"],"벤틀리":["BENTLEY"],"롤스로이스":["ROLLS ROYCE","ROLLS-ROYCE"],"폴스타":["POLESTAR"],"리비안":["RIVIAN"],"비엠더블유":["BMW","MINI"],"폭스바겐그룹":["VOLKSWAGEN","AUDI"],
    "지엠코리아":["CHEVROLET","GMC","BUICK","CADILLAC"],"지엠아시아":["CHEVROLET","GMC","BUICK","CADILLAC"],"폴스타오토모티브":["POLESTAR"],
    "폴스타오토모티브코리아":["POLESTAR"],"로터스카스":["LOTUS"],"두카티":["DUCATI"],"한국모터트레이딩":["HARLEY-DAVIDSON"]}
def norm(s): return re.sub(r"[^A-Z0-9]","",str(s).upper())
r["mk"]=r.make.str.strip().str.upper(); r["mdl"]=r.model.map(norm)
rows=[]
for _,x in ks.iterrows():
    makes=bm.get(str(x.brand).strip())
    if not makes: rows.append((x.recallSn,x.brand,"no-brand-map",np.nan)); continue
    toks={norm(t.split()[0]) for t in re.split(r"[,/]",str(x.modlNmInfo)) if t.strip()}
    toks={t for t in toks if len(t)>=2}
    c=r[r.mk.isin(makes)&(r.comp1==x.comp)&(r.date>=x.date-pd.Timedelta(days=1095))&(r.date<=x.date+pd.Timedelta(days=365))]
    c=c[[any(t in m or (len(m)>=2 and m in t) for t in toks) for m in c.mdl]]
    if len(c)==0: rows.append((x.recallSn,x.brand,"no-match",np.nan)); continue
    lag=(x.date-c.date).dt.days/30.44; i=lag.abs().idxmin()
    rows.append((x.recallSn,x.brand,"match",round(lag[i],1)))
M=pd.DataFrame(rows,columns=["recallSn","brand","status","lag_months_kr_minus_us"])
print("\n매칭 상태\n",M.status.value_counts().to_string())
m=M[M.status=="match"].lag_months_kr_minus_us
print(f"\n매칭 {len(m)}건: 미국이 먼저 {int((m>0.5).sum())}건, 비슷(±0.5개월) {int((m.abs()<=0.5).sum())}건, 한국이 먼저 {int((m<-0.5).sum())}건, 시차 중간값 {m.median():.1f}개월")
print("국내 수입 브랜드 vs 국산 매칭률:",M.assign(dom=M.brand.isin(["현대","기아","제네시스","케이지모빌리티","르노","르노삼성","지엠코리아","한국지엠"])).groupby("dom").status.apply(lambda s:round((s=="match").mean(),2)).to_dict())
print("브랜드별 매칭률\n",M.groupby("brand").status.apply(lambda s:(s=="match").mean()).sort_values(ascending=False).round(2).head(15).to_string())
k[["recallSn","brand","comp","date","emission"]].merge(M,on=["recallSn","brand"],how="left").to_csv(f"{d}/kr_vs_us_matches.csv",index=False,encoding="utf-8-sig")
dist.to_csv(f"{d}/kr_vs_us_component_share.csv",encoding="utf-8-sig")

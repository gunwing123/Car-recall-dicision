"""소비자24(국내) <-> NHTSA(미국) 매칭에 쓰는 공통 사전·함수 (kr_vs_us.py, us_model_to_kr.py, manual_verify.py가 공유).
이전에는 us_model_to_kr.py가 kr_vs_us.py 소스를 문자열로 잘라 eval()로 실행했음 -> 일반 import로 교체."""
import re, numpy as np, pandas as pd
bm={"벤츠":["MERCEDES-BENZ","MERCEDES BENZ"],"BMW":["BMW"],"미니":["MINI"],"혼다":["HONDA","ACURA"],"토요타":["TOYOTA"],"렉서스":["LEXUS"],
    "포드":["FORD"],"링컨":["LINCOLN"],"에프엘오토":["FORD","LINCOLN"],"폭스바겐":["VOLKSWAGEN"],"아우디":["AUDI"],"볼보":["VOLVO"],"지프":["JEEP"],
    "스텔란티스":["JEEP","CHRYSLER","DODGE","RAM","PEUGEOT"],"포르쉐":["PORSCHE"],"재규어랜드로버":["LAND ROVER","JAGUAR"],"랜드로버":["LAND ROVER"],
    "현대":["HYUNDAI","GENESIS"],"기아":["KIA"],"제네시스":["GENESIS"],"한국지엠":["CHEVROLET","GMC","BUICK","CADILLAC"],"쉐보레":["CHEVROLET"],
    "캐딜락":["CADILLAC"],"테슬라":["TESLA"],"닛산":["NISSAN"],"인피니티":["INFINITI"],"스바루":["SUBARU"],"마세라티":["MASERATI"],"페라리":["FERRARI"],
    "람보르기니":["LAMBORGHINI"],"벤틀리":["BENTLEY"],"롤스로이스":["ROLLS ROYCE","ROLLS-ROYCE"],"폴스타":["POLESTAR"],"리비안":["RIVIAN"],"비엠더블유":["BMW","MINI"],"폭스바겐그룹":["VOLKSWAGEN","AUDI"],
    "지엠코리아":["CHEVROLET","GMC","BUICK","CADILLAC"],"지엠아시아":["CHEVROLET","GMC","BUICK","CADILLAC"],"폴스타오토모티브":["POLESTAR"],
    "폴스타오토모티브코리아":["POLESTAR"],"로터스카스":["LOTUS"],"두카티":["DUCATI"],"한국모터트레이딩":["HARLEY-DAVIDSON"]}
rules=[("AIR BAGS",r"에어백"),("SEAT BELTS",r"안전띠|안전벨트|좌석벨트|시트벨트"),("BACK OVER PREVENTION",r"후방.{0,4}카메라|후방 영상|후방카메라"),
 ("SERVICE BRAKES",r"브레이크|제동"),("STEERING",r"조향|스티어링"),("FUEL SYSTEM",r"연료"),("POWER TRAIN",r"변속|트랜스미션|구동축|드라이브샤프트|파워트레인"),
 ("EXTERIOR LIGHTING",r"등화|램프|전조등|헤드라이트|라이트|방향지시"),("TIRES/WHEELS",r"타이어|휠"),("SUSPENSION",r"서스펜션|현가"),
 ("SEATS",r"좌석|시트"),("LATCHES/LOCKS",r"도어|잠금|래치|후드"),("VISIBILITY",r"와이퍼|유리|미러|거울"),
 ("ENGINE",r"엔진|크랭크|냉각"),("ELECTRICAL SYSTEM",r"배터리|고전압|전기|전원|소프트웨어|계기판|디스플레이|배선|퓨즈|컨트롤 ?유닛|ECU")]

def cat(s):
    for c,p in rules:
        if re.search(p,s): return c
    return "OTHER/STANDARD"
def norm(s): return re.sub(r"[^A-Z0-9]","",str(s).upper())
DOMESTIC={"현대","기아","제네시스","케이지모빌리티","르노","르노삼성","지엠코리아","한국지엠"}
COMP_MAP={"FUEL SYSTEM, GASOLINE":"FUEL SYSTEM","FUEL SYSTEM, DIESEL":"FUEL SYSTEM","FUEL SYSTEM, OTHER":"FUEL SYSTEM",
   "SERVICE BRAKES, HYDRAULIC":"SERVICE BRAKES","SERVICE BRAKES, AIR":"SERVICE BRAKES","SERVICE BRAKES, ELECTRIC":"SERVICE BRAKES",
   "WHEELS":"TIRES/WHEELS","TIRES":"TIRES/WHEELS","ENGINE AND ENGINE COOLING":"ENGINE","LATCHES/LOCKS/LINKAGES":"LATCHES/LOCKS","FORWARD COLLISION AVOIDANCE":"ELECTRICAL SYSTEM",
   "LANE DEPARTURE":"ELECTRICAL SYSTEM","ELECTRONIC STABILITY CONTROL (ESC)":"SERVICE BRAKES","HYBRID PROPULSION SYSTEM":"ELECTRICAL SYSTEM","COMMUNICATION":"ELECTRICAL SYSTEM"}

def load_kr(kr_path):
    """국내 리콜 로드: 날짜/브랜드/부품 분류/배출가스(환경부) 플래그"""
    k=pd.read_csv(kr_path,dtype=str,encoding="utf-8-sig")
    k["date"]=pd.to_datetime(k.recallPublictBgnde,format="%Y%m%d",errors="coerce")
    k=k[k.date>="2025-01-01"].copy()
    k["brand"]=k.productNm.str.extract(r"^\[(.+?)\]")[0].str.strip()
    k["defect"]=k.productNm.str.split(" - ",n=1).str[1].fillna("")
    k["txt"]=k.defect+" "+k.shrtcomCn.fillna("").str.replace(r"<[^>]+>"," ",regex=True)
    k["comp"]=[cat(s) for s in k.txt]
    k["emission"]=k.infoOriginInstt.str.contains("환경부",na=False)
    return k

def load_us_recalls(d):
    r=pd.read_csv(f"{d}/FLAT_RCL_POST_2010.txt",sep="\t",header=None,usecols=[1,2,3,4,6,10,15,19],
       names=["campno","make","model","year","comp","type","rcdate","desc_defect"],dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
    r=r[r.type=="V"].copy(); r["date"]=pd.to_datetime(r.rcdate,format="%Y%m%d",errors="coerce")
    r["comp1"]=r.comp.str.split(":").str[0].str.strip().str.upper().replace(COMP_MAP)
    r["mk"]=r.make.str.strip().str.upper(); r["mdl"]=r.model.map(norm)
    return r

def model_tokens(modlNmInfo):
    toks={norm(t.split()[0]) for t in re.split(r"[,/]",str(modlNmInfo)) if t.strip()}
    return {t for t in toks if len(t)>=2}

def match_us(x, r, past_days, fut_days):
    """국내 리콜 x와 같은 브랜드·차종 토큰·부품의 미국 리콜 후보 (창: 국내일 기준 -past_days ~ +fut_days). 후보 DataFrame 반환"""
    makes=bm.get(str(x.brand).strip())
    if not makes: return None
    toks=model_tokens(x.modlNmInfo)
    c=r[r.mk.isin(makes)&(r.comp1==x.comp)&(r.date>=x.date-pd.Timedelta(days=past_days))&(r.date<=x.date+pd.Timedelta(days=fut_days))]
    return c[[any(t in m or (len(m)>=2 and m in t) for t in toks) for m in c.mdl]]

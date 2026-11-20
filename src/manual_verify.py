"""한·미 매칭 수기 검증표 생성 (H4 86건 + H5 포착 건)
사용:  python src/manual_verify.py data/kr_recall_auto.csv data            -> data/manual_verification.csv 생성
       (엑셀로 열어 same_defect 열에 Y/N/? 입력)
       python src/manual_verify.py data/kr_recall_auto.csv data score      -> 입력한 결과로 진매칭률(precision) 계산
국내 결함 내용(소비자24 shrtcomCn)과 미국 DESC_DEFECT를 나란히 놓고 '같은 결함인가'를 사람이 판정한다."""
import pandas as pd, numpy as np, sys, os, re
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from kr_common import *
kr_path, d = sys.argv[1], sys.argv[2]; out=f"{d}/manual_verification.csv"
if len(sys.argv)>3 and sys.argv[3]=="score":
    M=pd.read_csv(out,encoding="utf-8-sig"); J=M[M.same_defect.isin(["Y","N","?"])]
    print(f"판정 {len(J)}/{len(M)}건 입력됨")
    for tag,sub in [("H4 매칭 전체",J),("H5 상위500 포착",J[J.h5_top500==1]),("H5 상위100 포착",J[J.h5_top100==1])]:
        if len(sub): print(f"  {tag}: 같은 결함 {int((sub.same_defect=='Y').sum())}/{len(sub)} = {(sub.same_defect=='Y').mean():.1%} (모름 {int((sub.same_defect=='?').sum())})")
    sys.exit()
k=load_kr(kr_path); ks=k[~k.emission].reset_index(drop=True); r=load_us_recalls(d)
def flags(K_):
    p=f"{d}/us_model_to_kr_top{K_}.csv"
    if not os.path.exists(p): return {}
    O=pd.read_csv(p); return {int(a):b for a,b in zip(O.recallSn,O.lead_months) if O.set_index("recallSn").loc[a,"status"]=="flagged"}
f100,f500=flags(100),flags(500)
clean=lambda s:re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",str(s)))[:350]
rows=[]
for x in ks.itertuples(index=False):
    sn=int(x.recallSn) if str(x.recallSn).isdigit() else x.recallSn
    c=match_us(x,r,round(36*30.44),round(12*30.44))
    in5=(sn in f500) or (sn in f100)
    if (c is None or len(c)==0) and not in5: continue          # 미국 매칭도 H5 포착도 없으면 검증 대상 아님
    row=dict(recallSn=x.recallSn,brand=x.brand,kr_date=x.date.date(),kr_comp=x.comp,kr_models=clean(x.modlNmInfo)[:120],kr_defect=clean(x.txt),
             us_campno="",us_date="",us_models="",us_desc_defect="",lag_months_kr_minus_us=np.nan,
             h5_top100=int(sn in f100),h5_top500=int(sn in f500),h5_lead_months=f500.get(sn,f100.get(sn,np.nan)),same_defect="",note="")
    if c is not None and len(c):
        lag=(x.date-c.date).dt.days/30.44; i=lag.abs().idxmin(); cc=c.loc[i]
        row.update(us_campno=cc.campno,us_date=cc.date.date(),us_models=", ".join(sorted(set(c[c.campno==cc.campno].model)))[:120],
                   us_desc_defect=clean(cc.desc_defect),lag_months_kr_minus_us=round(lag[i],1))
    rows.append(row)
M=pd.DataFrame(rows).sort_values(["h5_top500","h5_top100"],ascending=False)
M.to_csv(out,index=False,encoding="utf-8-sig")
print(f"검증 대상 {len(M)}건 저장 -> {out} | H5 포착 {int((M.h5_top500==1).sum())}건 포함 | same_defect 열에 Y/N/? 입력 후 'score' 모드 실행")

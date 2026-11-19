import pandas as pd, numpy as np, sys
d=sys.argv[1]; rng=np.random.default_rng(42)
c=pd.read_csv(f"{d}/FLAT_CMPL.txt",sep="\t",header=None,usecols=[3,4,5,11,15],names=["make","model","year","comp","datea"],dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
r=pd.read_csv(f"{d}/FLAT_RCL_POST_2010.txt",sep="\t",header=None,usecols=[1,2,3,4,6,10,15],names=["campno","make","model","year","comp","type","rcdate"],dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
r=r[r.type=="V"]
key=["make","model","year","comp"]
def norm(df,strict):
    for k in ["make","year"]: df[k]=df[k].str.strip().str.upper()
    df["model"]=df["model"].str.upper().str.strip()
    if strict: df["model"]=df["model"].str.replace(r"[^A-Z0-9]","",regex=True)
    df["comp"]=df["comp"].str.split(":").str[0].str.strip().str.upper()
for df in (c,r): norm(df,True)
c["d"]=pd.to_datetime(c.datea,format="%Y%m%d",errors="coerce"); r["d"]=pd.to_datetime(r.rcdate,format="%Y%m%d",errors="coerce")
c=c.dropna(subset=["d"]); r=r.dropna(subset=["d"]).drop_duplicates(key+["campno"])
c["m"]=c.d.dt.year*12+c.d.dt.month-1; r["m"]=r.d.dt.year*12+r.d.dt.month-1
cm=c[c.d>="2008-01-01"].groupby(key+["m"]).size().rename("n").reset_index()
ck=cm[key].drop_duplicates()
mt=r.merge(ck,on=key).shape[0]
print(f"[normalized model] recall units matched to complaint key: {mt}/{len(r)} = {mt/len(r):.1%}")
# treated: recall units whose key exists in complaints
T=r.merge(ck,on=key)[key+["m"]].reset_index(drop=True); T["g"]="recall"
# control: complaint keys never recalled (any campaign), model year in recall years
rk=r[key].drop_duplicates(); rk["_r"]=1
C=ck.merge(rk,on=key,how="left"); C=C[C._r.isna()].drop(columns="_r")
C=C[C.year.isin(T.year.unique())]
# pseudo date: sample recall month from treated units of same model year
pool=T.groupby("year").m.apply(np.array).to_dict()
C["m"]=[rng.choice(pool[y]) for y in C.year]; C["g"]="control"
print("treated units",len(T),"control keys",len(C))
U=pd.concat([T,C],ignore_index=True); U["uid"]=np.arange(len(U))
j=U.merge(cm,on=key,suffixes=("_e","")); j["off"]=j.m-j.m_e; j=j[j.off.between(-24,24)]
prof=j.groupby(["g","off"]).n.sum().unstack(0)/U.g.value_counts()
base=prof.loc[-24:-13].mean()
norm_=prof/base
out=pd.DataFrame({"recall_raw":prof["recall"],"control_raw":prof["control"],"recall_norm":norm_["recall"],"control_norm":norm_["control"]})
out.to_csv("/home/claude/data/control_profile.csv")
print(out.round(3).to_string())
# unit-level: growth ratio last 6m vs months -24..-13 (monthly avg)
p=j[j.off.between(-24,-1)].copy()
a=p[p.off>=-6].groupby("uid").n.sum()/6; b=p[p.off<=-13].groupby("uid").n.sum()/12
G=U.set_index("uid")[["g"]].join(a.rename("late")).join(b.rename("early")).fillna(0)
G=G[(G.late+G.early)>0]
G["up"]=G.late>G.early*1.5
print(G.groupby("g").agg(units=("up","size"),share_rising=("up","mean"),late=("late","mean"),early=("early","mean")).round(3))

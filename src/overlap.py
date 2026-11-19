import pandas as pd, numpy as np, sys
d=sys.argv[1]
c=pd.read_csv(f"{d}/FLAT_CMPL.txt",sep="\t",header=None,usecols=[3,4,5,11,15],names=["make","model","year","comp","datea"],
  dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
r=pd.read_csv(f"{d}/FLAT_RCL_POST_2010.txt",sep="\t",header=None,usecols=[1,2,3,4,6,10,15],names=["campno","make","model","year","comp","type","rcdate"],
  dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
print("complaints",len(c),"recalls",len(r))
r=r[r.type=="V"]
for df in (c,r):
    for k in ["make","model","year"]: df[k]=df[k].str.strip().str.upper()
    df["comp"]=df["comp"].str.split(":").str[0].str.strip().str.upper()
c["d"]=pd.to_datetime(c.datea,format="%Y%m%d",errors="coerce")
r["d"]=pd.to_datetime(r.rcdate,format="%Y%m%d",errors="coerce")
c=c[c.d>="2008-01-01"].dropna(subset=["d"]); r=r.dropna(subset=["d"])
key=["make","model","year","comp"]
r=r.drop_duplicates(key+["campno"])
c["m"]=c.d.dt.year*12+c.d.dt.month-1
r["m"]=r.d.dt.year*12+r.d.dt.month-1
print("complaints 2008+",len(c),"recall units (campaign x make/model/year/comp)",len(r))
cm=c.groupby(key+["m"]).size().rename("n").reset_index()
j=r[key+["campno","m"]].merge(cm,on=key,suffixes=("_r",""))
j["off"]=j["m"]-j["m_r"]
j=j[j.off.between(-24,24)]
units=len(r)
matched=r.merge(cm[key].drop_duplicates(),on=key).shape[0]
print(f"recall units with any complaint on same key: {matched}/{units} = {matched/units:.1%}")
pre=j[j.off.between(-12,-1)].groupby(["campno"]+key).n.sum()
print(f"units with >=1 complaint in 12m BEFORE recall: {len(pre)}/{units} = {len(pre)/units:.1%}")
prof=j.groupby("off").n.sum()/units
prof.to_csv("/home/claude/data/event_profile.csv")
print(prof.round(3).to_string())
# top example: biggest pre-recall buildup
g=j.groupby(["campno"]+key)
s=pd.DataFrame({"pre12":j[j.off.between(-12,-1)].groupby(["campno"]+key).n.sum(),
                "pre13_24":j[j.off.between(-24,-13)].groupby(["campno"]+key).n.sum()}).fillna(0)
s["ratio"]=(s.pre12+1)/(s.pre13_24+1)
print(s.sort_values("pre12",ascending=False).head(8).to_string())

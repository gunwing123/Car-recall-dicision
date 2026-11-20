import pandas as pd, numpy as np, sys, time
d=sys.argv[1]; t0=time.time()
END_T=2026*12+3-1          # last prediction month (2026-03) so t+6 is observed
START_T=2010*12+0          # recalls file starts 2010
H=6                        # label horizon (months)
key=["make","model","year","comp"]
def norm(df):
    for k in ["make","year"]: df[k]=df[k].str.strip().str.upper()
    df["model"]=df["model"].str.upper().str.replace(r"[^A-Z0-9]","",regex=True)
    df["comp"]=df["comp"].str.split(":").str[0].str.strip().str.upper()
c=pd.read_csv(f"{d}/FLAT_CMPL.txt",sep="\t",header=None,usecols=[1,3,4,5,6,8,9,10,11,15],
    names=["odino","make","model","year","crash","fire","inj","dth","comp","datea"],dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
norm(c)
n0=len(c); c=c[~(c.duplicated(["odino","make","model","year","comp"])&c.odino.notna())]  # 한 불만(ODINO)에 같은 상위 부품이 여러 하위 부품으로 등록된 중복 제거
print("complaint rows dedup by ODINO:",n0,"->",len(c))
c["dt"]=pd.to_datetime(c.datea,format="%Y%m%d",errors="coerce")
c=c.dropna(subset=["dt"]); c=c[c.dt>="2008-01-01"]
c=c[c.year.str.fullmatch(r"(19|20)\d\d")]
c["m"]=(c.dt.dt.year*12+c.dt.dt.month-1).astype(np.int32)
c["crash"]=(c.crash=="Y").astype(np.int8); c["fire"]=(c.fire=="Y").astype(np.int8)
c["inj"]=(pd.to_numeric(c.inj,errors="coerce").fillna(0)>0).astype(np.int8)
c["dth"]=(pd.to_numeric(c.dth,errors="coerce").fillna(0)>0).astype(np.int8)
c=c.drop(columns=["datea","dt"])
print("complaints",len(c),f"{time.time()-t0:.0f}s")

r=pd.read_csv(f"{d}/FLAT_RCL_POST_2010.txt",sep="\t",header=None,usecols=[1,2,3,4,6,10,13,15],
    names=["campno","make","model","year","comp","type","infl","rcdate"],dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip")
r=r[r.type=="V"]; norm(r)
r["dt"]=pd.to_datetime(r.rcdate,format="%Y%m%d",errors="coerce"); r=r.dropna(subset=["dt"])
r["m"]=(r.dt.dt.year*12+r.dt.dt.month-1).astype(np.int32)
r=r.drop_duplicates(key+["campno"])
print("recall units",len(r)); print("INFLUENCED_BY:\n",r.drop_duplicates("campno").infl.str.strip().value_counts(dropna=False).head(6).to_string())

cm=c.groupby(key+["m"],observed=True).agg(n=("crash","size"),crash=("crash","sum"),fire=("fire","sum"),inj=("inj","sum"),dth=("dth","sum")).reset_index()
del c
cm["kid"]=pd.MultiIndex.from_frame(cm[key]).factorize()[0] if False else cm.groupby(key,sort=False).ngroup()
K=cm[["kid"]+key].drop_duplicates("kid").set_index("kid")
print("keys",len(K))
rk=r.merge(K.reset_index(),on=key)[["kid","m","campno"]]
print("recall units on complaint keys",len(rk))
rk.to_pickle(f"{d}/recall_units.pkl")

# group-level monthly counts for sibling features
cm["mmc"]=cm.groupby(["make","model","comp"],sort=False).ngroup()
cm["mc"]=cm.groupby(["make","comp"],sort=False).ngroup()
K=K.join(cm.drop_duplicates("kid").set_index("kid")[["mmc","mc"]])
K.to_pickle(f"{d}/keys.pkl")

def roll12(df,g):
    a=df.groupby([g,"m"]).n.sum().reset_index()
    out=[]
    for s in range(12):
        b=a.copy(); b["m"]=b.m+s; out.append(b)
    return pd.concat(out).groupby([g,"m"]).n.sum().rename(f"{g}_n12")
mmc12=roll12(cm,"mmc"); mc12=roll12(cm,"mc")
# sibling recalls: recalls known (date<=t) on same make-model-comp, other years, last 12m
r2=r.merge(K.reset_index()[["kid","make","model","year","comp","mmc"]],on=key,how="inner")
ra=r2.groupby(["mmc","m"]).campno.nunique().reset_index()
out=[]
for s in range(12):
    b=ra.copy(); b["m"]=b.m+s; out.append(b)
mmc_rcl12=pd.concat(out).groupby(["mmc","m"]).campno.sum().rename("mmc_rcl12")

first=cm.groupby("kid").m.min(); last=cm.groupby("kid").m.max()
kids=first[first<=END_T].index.values
parts=[]
for ch in np.array_split(kids,40):
    f=first.loc[ch].values; l=np.minimum(last.loc[ch].values+12,END_T)
    L=l-f+1
    kk=np.repeat(ch,L); mm=np.concatenate([np.arange(a,b+1) for a,b in zip(f,l)]).astype(np.int32)
    p=pd.DataFrame({"kid":kk,"m":mm})
    p=p.merge(cm[cm.kid.isin(ch)][["kid","m","n","crash","fire","inj","dth"]],on=["kid","m"],how="left").fillna(0)
    p=p.sort_values(["kid","m"]).reset_index(drop=True)
    g=p.groupby("kid")
    for col in ["n","crash","fire","inj","dth"]:
        cs=g[col].cumsum(); p[f"{col}_cum"]=cs
        for w in ([1,3,6,12,24] if col=="n" else [12]):
            if w==1: continue
            p[f"{col}{w}"]=cs-cs.groupby(p.kid).shift(w).fillna(0)
    p["n_prev6"]=p.n12-p.n6; p["n_prev12"]=p.n24-p.n12
    p["mon_since_first"]=p.m-p.kid.map(first)
    p=p[(p.n12>0)&(p.m>=START_T)&(p.m<=END_T)]
    # recall events on this key
    rr=rk[rk.kid.isin(ch)][["kid","m"]].drop_duplicates()
    if len(rr):
        # label: recall in (t, t+H]; exclusion: recall in [t-11, t]
        lab=[];exc=[]
        for s in range(1,H+1):
            b=rr.copy(); b["m"]=b.m-s; lab.append(b)
        for s in range(0,12):
            b=rr.copy(); b["m"]=b.m+s; exc.append(b)
        lab=pd.concat(lab).drop_duplicates(); lab["y"]=1
        exc=pd.concat(exc).drop_duplicates(); exc["post"]=1
        p=p.merge(lab,on=["kid","m"],how="left").merge(exc,on=["kid","m"],how="left")
        # prior recalls on key before t
        pr=rr.groupby("kid").m.apply(np.sort).to_dict()
        p["prior_rcl"]=[np.searchsorted(pr.get(k,np.array([],int)),t,side="right") for k,t in zip(p.kid,p.m)]
    else:
        p["y"]=np.nan; p["post"]=np.nan; p["prior_rcl"]=0
    p["y"]=p.y.fillna(0).astype(np.int8); p=p[p.post.isna()].drop(columns="post")
    parts.append(p)
P=pd.concat(parts,ignore_index=True); del parts
P=P.join(K[["make","year","comp","mmc","mc"]],on="kid")
P["age"]=P.m//12-P.year.astype(int)
P=P.join(mmc12,on=["mmc","m"]).join(mc12,on=["mc","m"]).join(mmc_rcl12,on=["mmc","m"])
P["sib_n12"]=P.mmc_n12-P.n12
P["mmc_rcl12"]=P.mmc_rcl12.fillna(0)
P=P.drop(columns=["mmc_n12","year"])
for col in P.columns:
    if P[col].dtype==np.float64: P[col]=P[col].astype(np.float32)
P.to_pickle(f"{d}/panel.pkl")
print("panel rows",len(P),"positives",int(P.y.sum()),f"rate {P.y.mean():.4%}",f"{time.time()-t0:.0f}s")
print(P.groupby(P.m//12).y.agg(["size","sum","mean"]).to_string())

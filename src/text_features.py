import pandas as pd, numpy as np, sys, time, gc, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
warnings.filterwarnings("ignore")
d=sys.argv[1]; t0=time.time(); NT=12
key=["make","model","year","comp"]
K=pd.read_pickle(f"{d}/keys.pkl").reset_index()[["kid"]+key]
kw={"kw_stall":r"\bSTALL|SHUT(S|TING)? (OFF|DOWN)|ENGINE (DIED|DIES|QUIT)|LOSS OF (MOTIVE )?POWER|LOST (ALL )?POWER",
    "kw_fire":r"\bFIRE\b|SMOKE|BURN|FLAME|MELT",
    "kw_nodeploy":r"(AIR ?BAGS?) (DID NOT|DIDN'T|FAILED TO|DIDNT) DEPLOY",
    "kw_brake_fail":r"BRAKES? (FAILED|FAILURE|WENT TO THE FLOOR|DID NOT WORK)|NO BRAKES",
    "kw_steer_loss":r"(LOST|LOSS OF) (POWER )?STEERING|STEERING (LOCKED|FAILED)",
    "kw_sudden_accel":r"SUDDEN(LY)? ACCELERAT|UNINTENDED ACCELERAT|SURGED",
    "kw_warning_light":r"WARNING LIGHT|CHECK ENGINE|DASH(BOARD)? LIGHT|INDICATOR",
    "kw_while_driving":r"WHILE DRIVING|AT (\d+ )?MPH|ON THE (HIGHWAY|FREEWAY|INTERSTATE)",
    "kw_repeat":r"AGAIN|SECOND TIME|THIRD TIME|REPEATED|MULTIPLE TIMES|SEVERAL TIMES",
    "kw_dealer_noremedy":r"(DEALER|DEALERSHIP).{0,60}(COULD NOT|COULDN'T|UNABLE|NO FIX|NOT FIX|NOT DUPLICATE)",
    "kw_known_issue":r"KNOWN (ISSUE|PROBLEM|DEFECT)|COMMON (ISSUE|PROBLEM)|OTHER OWNERS|MANY OWNERS"}
STRIP=r"\bRECALL\w*|\bCAMPAIGN\w*|\bNHTSA\b|TL\*|\*[A-Z]{2}\b"
def chunks():
    for c in pd.read_csv(f"{d}/FLAT_CMPL.txt",sep="\t",header=None,usecols=[3,4,5,11,15,19],names=["make","model","year","comp","datea","txt"],
                         dtype=str,encoding="latin-1",quoting=3,on_bad_lines="skip",chunksize=200000):
        for k in ["make","year"]: c[k]=c[k].str.strip().str.upper()
        c["model"]=c["model"].str.upper().str.replace(r"[^A-Z0-9]","",regex=True)
        c["comp"]=c["comp"].str.split(":").str[0].str.strip().str.upper()
        dt=pd.to_datetime(c.datea,format="%Y%m%d",errors="coerce")
        c["m"]=dt.dt.year*12+dt.dt.month-1
        c=c[(dt>="2008-01-01")&c.year.fillna("").str.fullmatch(r"(19|20)\d\d")]
        c=c.merge(K,on=key,how="inner")[["kid","m","txt"]]
        c["txt"]=c.txt.fillna("").str.upper().str.slice(0,1500)
        c["mention_recall"]=c.txt.str.contains(r"RECALL|CAMPAIGN",regex=True).astype(np.int8)
        c["txt"]=c.txt.str.replace(STRIP," ",regex=True)
        yield c
rng=np.random.default_rng(0); flags=[]; samp=[]
for c in chunks():
    for k,p in kw.items(): c[k]=c.txt.str.contains(p,regex=True).astype(np.int8)
    pre=c[c.m<2020*12]; samp.append(pre.txt[rng.random(len(pre))<0.12])
    flags.append(c.drop(columns="txt"))
F=pd.concat(flags,ignore_index=True); S=pd.concat(samp); del flags,samp; gc.collect()
print("complaints",len(F),"recall-mention share",round(F.mention_recall.mean(),3),"sample",len(S),f"{time.time()-t0:.0f}s",flush=True)
print("keyword share:",F[list(kw)].mean().round(3).to_dict(),flush=True)
vec=TfidfVectorizer(max_features=8000,min_df=20,max_df=0.4,stop_words="english",token_pattern=r"(?u)\b[A-Z]{3,}\b",lowercase=False)
X=vec.fit_transform(S); del S
nmf=NMF(n_components=NT,random_state=0,init="nndsvda",max_iter=300).fit(X); del X; gc.collect()
voc=np.array(vec.get_feature_names_out())
tops={f"tp{i:02d}":" ".join(voc[np.argsort(-nmf.components_[i])[:8]]) for i in range(NT)}
for k,v in tops.items(): print(k,v,flush=True)
pd.Series(tops).to_csv(f"{d}/topics.csv")
lab=[]
for c in chunks():
    if len(c): lab.append(nmf.transform(vec.transform(c.txt)).argmax(1).astype(np.int8))
F["topic"]=np.concatenate(lab)
print("topic shares",F.topic.value_counts(normalize=True).sort_index().round(3).to_dict(),f"{time.time()-t0:.0f}s",flush=True)
for i in range(NT): F[f"tp{i:02d}"]=(F.topic==i).astype(np.int8)
cols=list(kw)+[f"tp{i:02d}" for i in range(NT)]+["mention_recall"]
km=F.groupby(["kid","m"])[cols].sum(); km["ntxt"]=F.groupby(["kid","m"]).size(); km=km.reset_index(); del F; gc.collect()
parts=[]
for s in range(12):
    b=km.copy(); b["m"]=b.m+s; parts.append(b)
R=pd.concat(parts).groupby(["kid","m"])[cols+["ntxt"]].sum(); del parts
for col in cols: R[col]=(R[col]/R.ntxt).astype(np.float32)
R=R.drop(columns="ntxt").add_suffix("_r12").reset_index()
R.to_pickle(f"{d}/text_feats.pkl"); print("text feature rows",len(R),f"{time.time()-t0:.0f}s",flush=True)

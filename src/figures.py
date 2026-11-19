"""발표용 그래프 (results/ 및 data/ 산출물 사용) -> figures/*.png"""
import pandas as pd, numpy as np, json, sys, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
d=sys.argv[1]; out=sys.argv[2]; os.makedirs(out,exist_ok=True)
for f in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family":"Noto Sans CJK JP","axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#b4b2a9",
    "axes.labelcolor":"#3d3d3a","xtick.color":"#73726c","ytick.color":"#73726c","axes.grid":True,"grid.color":"#ebeae4","grid.linewidth":0.8,
    "axes.axisbelow":True,"font.size":11,"axes.titlesize":13,"axes.titleweight":"bold","figure.dpi":150,"lines.linewidth":2})
B,O,A,G,INK="#2a78d6","#eb6834","#1baf7a","#a3a29b","#3d3d3a"
def save(fig,name): fig.tight_layout(); fig.savefig(f"{out}/{name}",bbox_inches="tight",facecolor="white"); plt.close(fig)

# 1. event study
c=pd.read_csv(f"{d}/control_profile.csv",index_col=0).loc[-24:12]
fig,ax=plt.subplots(figsize=(8,4.2))
ax.axvspan(0,12,color="#f6f5f1"); ax.text(11.6,c.recall_norm.max()*0.97,"리콜 이후\n(누수 구간, 피처에서 제외)",ha="right",va="top",color="#73726c",fontsize=9)
ax.plot(c.index,c.recall_norm,color=B,marker="o",ms=3); ax.plot(c.index,c.control_norm,color=G,marker="o",ms=3)
ax.axvline(0,color=INK,lw=1,ls="--")
ax.annotate(f"리콜된 차종-부품 {c.recall_norm.loc[-1]:.2f}배",(-1,c.recall_norm.loc[-1]),xytext=(-15,2.4),color=B,fontsize=10,arrowprops=dict(arrowstyle="-",color=B,lw=0.8))
ax.annotate(f"리콜 없는 차종-부품 {c.control_norm.loc[-1]:.2f}배",(-1,c.control_norm.loc[-1]),xytext=(-20,1.75),color="#5f5e5a",fontsize=10,arrowprops=dict(arrowstyle="-",color=G,lw=0.8))
ax.set(xlabel="리콜(비교 그룹은 가상 기준일) 대비 개월",ylabel="상대 불만 수준 (24~13개월 전 = 1)",title="리콜 직전 불만이 비교 그룹보다 가파르게 늘어난다 (H1)")
save(fig,"fig1_event_study.png")

# 2. model 1 comparison (P@100)
r=json.load(open(f"{d}/results.json")); base=0.0086
names=["무작위","규칙\n(최근 3개월 불만)","LightGBM","로지스틱 회귀"]; vals=[base,r["rule_n3"]["P@100"],r["lgb_full"]["P@100"],r["logistic"]["P@100"]]
fig,ax=plt.subplots(figsize=(7,4))
bars=ax.bar(names,[v*100 for v in vals],color=[G,G,G,B],width=0.55)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v*100+0.2,f"{v*100:.1f}%",ha="center",color=INK,fontsize=10)
ax.set(ylabel="월 상위 100개 중 실제 리콜 비율 (%)",title="리콜 조기경보: 로지스틱이 무작위의 약 11배 (테스트 2022~2026)"); ax.grid(axis="x",visible=False)
save(fig,"fig2_model1_precision.png")

# 3. drift: retraining strategy
dr=pd.read_csv(f"{d}/drift_results.csv")
fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,m,t in zip(axs,["lightgbm","logistic"],["LightGBM","로지스틱 회귀"]):
    for sc,col,lab in [("fixed (2010-2019)",G,"고정 학습 (2010~2019)"),("sliding 5y",B,"매년 재학습 (최근 5년)")]:
        s=dr[(dr.model==m)&(dr.scheme==sc)].sort_values("test_year")
        ax.plot(s.test_year,s.p_at_100*100,color=col,marker="o",ms=5,label=lab)
        ax.text(s.test_year.iloc[-1]+0.08,s.p_at_100.iloc[-1]*100,f"{s.p_at_100.mean()*100:.1f}%\n(평균)",color=col,fontsize=9,va="center")
    ax.set(title=t,xticks=[2022,2023,2024,2025],xlim=(2021.8,2025.6),xlabel="테스트 연도")
axs[0].set_ylabel("월 상위 100개 정답률 (%)"); axs[0].legend(frameon=False,loc="upper left",fontsize=9.5)
fig.suptitle("패턴이 해마다 바뀐다: 재학습하면 LightGBM 정답률이 2.7% → 8.2%",fontweight="bold",fontsize=13)
save(fig,"fig3_drift_retraining.png")

# 4. component recall rate drift
P=pd.read_pickle(f"{d}/panel.pkl"); P["yr"]=P.m//12
comp={"AIR BAGS":("에어백",O,0.25),"ELECTRICAL SYSTEM":("전기계통",B,0),"STRUCTURE":("차체 구조",A,-0.25)}
fig,ax=plt.subplots(figsize=(8,4))
for k,(lab,col,dy) in comp.items():
    s=P[(P.comp==k)&(P.yr<=2025)].groupby("yr").y.mean()*100
    ax.plot(s.index,s.values,color=col,marker="o",ms=3); ax.text(s.index[-1]+0.2,s.values[-1]+dy,lab,color=col,va="center",fontsize=10)
ax.set(xlabel="연도",ylabel="6개월 내 리콜 비율 (%)",title="부품별 리콜 비율의 변화 (개념 드리프트의 근거)",xlim=(2009.5,2026.8))
save(fig,"fig4_component_drift.png")

# 5. KR vs US lag
M=pd.read_csv(f"{d}/kr_vs_us_matches.csv"); m=M[M.status=="match"].lag_months_kr_minus_us.clip(-3,18)
fig,ax=plt.subplots(figsize=(7.5,4))
ax.hist(m,bins=np.arange(-3,19,1),color=B,edgecolor="white",linewidth=2)
ax.axvline(0,color=INK,lw=1,ls="--"); ax.axvline(m.median(),color=O,lw=1.5)
ax.text(m.median()+0.3,ax.get_ylim()[1]*0.9,f"중간값 {m.median():.0f}개월",color=O,fontsize=10)
ax.text(-2.8,ax.get_ylim()[1]*0.9,"← 한국이 먼저",color="#73726c",fontsize=9.5); ax.text(9,ax.get_ylim()[1]*0.75,"미국이 먼저 →",color="#73726c",fontsize=9.5)
n=len(m); us=int((m>0.5).sum())
ax.set(xlabel="국내 리콜 공표일 − 미국 리콜 접수일 (개월, 18 이상은 18로 표시)",ylabel="리콜 수",
       title=f"같은 차종·부품 리콜 {n}건 중 {us}건({us/n:.0%})은 미국이 먼저 (H4)"); ax.grid(axis="x",visible=False)
save(fig,"fig5_kr_us_lag.png")

# 6. text H2
t=json.load(open(f"{d}/text_results.json"))
lab={"base (no text)":"텍스트 없음","+ keywords":"+ 키워드","+ keywords + topics":"+ 키워드 + 토픽"}
fig,ax=plt.subplots(figsize=(7,3.2))
for i,(k,l) in enumerate(lab.items()):
    v=t[k]["test PR-AUC"]; lo,hi=map(float,t[k]["test PR-AUC 95% CI"].split("-"))
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color=B if i==0 else G,ms=8,capsize=4,lw=2); ax.text(hi+0.0006,i,f"{v:.4f}",va="center",fontsize=10,color=INK)
ax.set_yticks(range(3)); ax.set_yticklabels(list(lab.values())); ax.invert_yaxis()
ax.set(xlabel="테스트 PR-AUC (95% 신뢰구간)",title="불만 텍스트를 넣어도 예측이 좋아지지 않는다 (H2)",xlim=(0.033,0.054))
save(fig,"fig6_text_h2.png")
print(sorted(os.listdir(out)))

# 7. H5: US model flags Korean imported-car recalls
rows=[]
for K_ in [100,500]:
    O=pd.read_csv(f"{d}/us_model_to_kr_top{K_}.csv"); ev=O[O.status!="US key not in panel"]; rows.append((K_,(ev.status=="flagged").mean()))
base={100:0.020,500:0.069}
fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(2); w=0.34
b1=ax.bar(x-w/2,[base[k]*100 for k,_ in rows],w,color=G,label="비교 기준: 임의의 미국 차종-부품")
b2=ax.bar(x+w/2,[v*100 for _,v in rows],w,color=B,label="국내 리콜이 난 수입차 차종-부품")
for bs in (b1,b2):
    for b in bs: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.6,f"{b.get_height():.1f}%",ha="center",fontsize=10,color=INK)
ax.set_xticks(x); ax.set_xticklabels(["미국 월 상위 100개","미국 월 상위 500개"])
ax.set(ylabel="국내 공표 전 12개월 안에 경보된 비율 (%)",title="미국 조기경보 모델이 국내 수입차 리콜을 미리 잡는다 (H5)")
ax.legend(frameon=False,fontsize=9.5,loc="upper left"); ax.grid(axis="x",visible=False)
save(fig,"fig7_us_model_to_kr.png")
print("fig7 done")

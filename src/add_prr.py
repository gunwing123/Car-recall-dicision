"""PRR(비례보고비율) 피처: 판매량을 모르는 상태에서 '인기 차종이라 모든 부품 불만이 많은 경우'와 '특정 부품만 비정상 급증한 경우'를 분리.
PRR(v,c,t) = [n(v,c,t)/sum_c' n(v,c',t)] / [sum_v' n(v',c,t)/sum_{v',c'} n(v',c',t)],  v=(제조사,차종,연식), n=최근 12개월 불만 수(n12)
주의: 패널은 '리콜 직후 12개월 제외'로 일부 행이 빠져 있어 분모(차종 전체/시장 전체)가 약간 과소 추정됨 -> 근사치.  사용: python src/add_prr.py data"""
import pandas as pd, numpy as np, sys
d=sys.argv[1]; P=pd.read_pickle(f"{d}/panel.pkl"); K=pd.read_pickle(f"{d}/keys.pkl")
P=P.join(K[["model","year"]].rename(columns={"model":"md_","year":"yr_"}),on="kid")
P["mk_"]=P.make
P["vid"]=P.groupby(["mk_","md_","yr_"],sort=False).ngroup()
tot_v=P.groupby(["vid","m"]).n12.transform("sum"); tot_c=P.groupby(["comp","m"]).n12.transform("sum"); tot=P.groupby("m").n12.transform("sum")
P["prr12"]=((P.n12/tot_v)/(tot_c/tot)).astype(np.float32)
P=P.drop(columns=["mk_","md_","yr_","vid"]); P.to_pickle(f"{d}/panel_prr.pkl"); print("saved panel_prr.pkl",P.prr12.describe().round(3).to_dict())

# Car-recall-dicision
산업데이터 분석 팀플 - 리콜 예측 및 결정


미국 NHTSA 고객 불만(Complaints)과 리콜(Recalls) 공개 데이터로
(1) 리콜 직전 불만 신호가 있는지, (2) 6개월 안에 리콜될 차종-부품을 미리 고를 수 있는지,
(3) 리콜이 다른 연식으로 확대될지 예측할 수 있는지를 분석합니다.

## 데이터 (저장소에 포함하지 않음)
`data/` 폴더에 아래 파일을 내려받아 압축을 풉니다.
- 불만: https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip (컬럼 설명: CMPL.txt)
- 리콜: https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip (컬럼 설명: RCL.txt)

## 실행
```bash
pip install pandas numpy scikit-learn lightgbm
python src/overlap.py data        # 리콜 전후 불만 추이
python src/control.py data        # 리콜 없는 차종-부품과 비교
python src/build_panel.py data    # 차종-연식-부품 x 월 패널 + 6개월 리콜 라벨
python src/train.py data          # 규칙 / 로지스틱 / LightGBM
python src/expansion.py data      # 리콜 범위 확대 예측
python src/text_features.py data  # 불만 텍스트 키워드·토픽 (메모리 약 4GB)
python src/train_text.py data     # 텍스트 추가 효과(H2) 검증
```

## 설정
- 키: 제조사 + 차종(영숫자만) + 연식 + 부품 상위 분류(COMPDESC 콜론 앞)
- 라벨: 다음 6개월 안에 같은 키로 리콜 접수(RCDATE)
- 대상: 최근 12개월 불만 1건 이상인 키-월. 리콜 직후 12개월은 제외(누수 방지)
- 분할: 학습 2010~2019 / 검증 2020~2021 / 테스트 2022.01~2026.03

## 주요 결과 (results/)
| 분석 | 결과 |
|---|---|
| 리콜 전 불만 추이 | 리콜 직전 달 불만이 2년 전 대비 1.94배 (리콜 없는 차종 1.36배) |
| 리콜 예측 (로지스틱) | 테스트 PR-AUC 0.043 (기본 0.0086), 월 상위 100개 정답률 9.5% |
| 리콜 확대 예측 (로지스틱) | PR-AUC 0.171 (기본 0.087), 상위 20%로 확대 리콜의 44% 포착 |
| 텍스트 추가 (H2) | 개선 없음 (PR-AUC 0.0429 → 0.0431, 신뢰구간 겹침) |

## 한계
- 불만과 같은 키로 연결되는 리콜은 약 25%
- 판매량 보정 없음
- 확대 라벨은 부품 상위 분류 기준이라 무관한 결함이 섞일 수 있음

##  CNAN 기반 추천 시스템

이 프로젝트는 **CNAN (Cross Neighborhood Attention Network)** 기반의 추천 시스템 모델을 구현하고, [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) 데이터셋을 바탕으로 평가합니다. 이웃 기반의 상호작용 정보를 효과적으로 활용하기 위해 **1-hop / 2-hop** 이웃 임베딩을 추출하여 어텐션 기반으로 학습합니다.

---

###  프로젝트 구조

```
CNAN/
├── dataset.py                      # 데이터 로딩 및 전처리
├── model.py                        # CNAN 모델 정의
├── train_with_precompute.py       # 1. 이웃 정보 계산 + 학습
├── train_from_cache.py            # 2. 저장된 이웃 정보로 학습
├── precomputed_and_eval.py        # 3. 이웃 계산 + 평가
├── eval_from_cache.py             # 4. 저장된 이웃 정보로 평가
├── plot.py                        # 학습 손실 및 평가 결과 시각화

```

---

###  실행 방법

#### 1. 데이터 준비

```bash
# MovieLens 100K 다운로드 후 u.data 경로를 dataset.py에서 설정하거나 아래에 넣어주세요
ml-100k/u.data
```

#### 2. 이웃 정보 계산 + 학습 (최초 실행)

```bash
python train_with_precompute.py
```

#### 3. 저장된 이웃 정보로 빠르게 학습 재실행

```bash
python train_from_cache.py
```

#### 4. 이웃 정보 계산 + 평가

```bash
python precomputed_and_eval.py
```

#### 5. 저장된 이웃 정보로 빠르게 평가 재실행

```bash
python eval_from_cache.py
```

---

###  성능 시각화

학습 결과를 아래 명령어로 시각화할 수 있습니다:

```bash
python plot.py
```

생성되는 그래프:

* `loss_curve.png` – 학습 손실
* `loss_lr_curve.png` – 손실 + 러닝레이트 변화
* `eval_metrics.png` – RMSE / MAE 변화
* `pred_vs_true.png` – 예측값 vs 실제값 산점도

---

###  주요 설정

| 설정 항목     | 값              |
| --------- | -------------- |
| 임베딩 차원    | 32             |
| 배치 크기     | 128            |
| 학습률       | 0.001          |
| scheduler | StepLR(step=3) |
| 평가 지표     | RMSE, MAE      |
| 장치        | CUDA or CPU    |

---

###  연구 목적

* 이 모델은 사용자와 아이템 간 직접적/간접적 이웃 관계를 학습에 반영하여 **구조적 추천을 강화**합니다.

---

###  참고 논문

* CNAN 모델 구조는 "task-oriented collaborative graph embedding using explicit high-order proximity for recommendation" 논문 구조를 참고함


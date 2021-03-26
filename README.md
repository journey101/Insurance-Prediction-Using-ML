# dsft01-section2-solo-project

### 1. 문제정의 및 프로젝트 목적 소개
- 본 프로젝트의 목적은 미국의 한 건강보험 가입자 38만여명의 고객데이터를 바탕으로 아래의 2가지 솔루션을 찾는 것에 있습니다.
- 머신러닝 예측모델로 차 보험 가입 가능성이 높은 고객 DB 추출
- 선별된 고객들의 특성 중 집중할 특성을 선별하여 마케팅 전략 도출

### 2. 데이터 분석 및 예측모델 설명
먼저, 사용할 데이터에 대하여 간략히 설명하겠습니다.
사용한 데이터는 미국의 한 건강보험 가입자 데이터로 데이터 특성 중 'Response(차 보험에 대한 긍정 여부, 1=긍정)' 가 최종적으로 예측하고자 하는 타겟 변수입니다.
※데이터 출처: https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction?select=train.csv

> 분석에 앞서, 아래 2가지 필터로 데이터를 가공하였습니다.
- 1) 원본 데이터 특성 중 'Driving License(운전면허 보유여부)'데이터에서 운전면허가 있는 고객(1값)으로 1차 선별.
- 2) 'Previously_Insured(차 보험 보유 여부)' 에서 차 보험이 없는 고객 (0값) 으로 2차 선별.

> 머신러닝 예측모델 중에서는 성능이 좋다고 알려진 'Random Forest', 'lightGBM', 'Catboost'모델을 비교하였습니다.
아래는 각 모델의 성능비교표입니다.
참고로, 각 모델의 성능을 가장 높일 수 있는 optimal한 최적 임계치를 적용한 수치입니다.
![image](https://user-images.githubusercontent.com/70046278/112586169-9e4e9a00-8e3e-11eb-883a-02ae5eb26ca8.png)

> 3가지 머신러닝 중 베스트 모델로는 catboost 모델을 선정했습니다.
선정 이유는 모델의 정확도(accuracy score)는 가장 낮지만, 과적합을 최소화하면서 테스트데이터의 예측도는 높일 수 있는 AUC score가 가장 높고, 타겟에 대한 예측도인 f1-score 가 높다는 점을 고려했습니다.
추가로 본 데이터의 특성 중 '지역코드'와 '판매채널' 데이터의 Cardinality(범주형 데이터의 수)가 높다는 점에서 Randomforest와 lightGBM 모델에 비하여 Catboost모델의 속도가 빠르다는 장점도 고려했습니다.

### 3. 인사이트 및 결론
※블로그 참고: https://medium.com/p/345c871239d/edit

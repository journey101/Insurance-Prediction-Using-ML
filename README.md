# Insurance-Prediction-Using-Machine-Learning

### 1. 문제정의 및 프로젝트 목적 소개
본 프로젝트의 목적은 미국의 한 건강보험 가입자 38만여명의 고객데이터를 바탕으로 아래의 2가지 솔루션을 찾는 것에 있습니다.
- 머신러닝 예측모델로 차 보험 가입 가능성이 높은 고객 DB 추출
- 선별된 고객들의 특성 중 집중할 특성을 선별하여 마케팅 전략 도출

### 2. 데이터 분석 및 예측모델 설명
먼저, 사용할 데이터에 대하여 간략히 설명하겠습니다.
사용한 데이터는 미국의 한 건강보험 가입자 데이터로 데이터 특성 중 'Response(차 보험에 대한 긍정 여부, 1=긍정)' 가 최종적으로 예측하고자 하는 타겟 변수입니다.

※[데이터 출처](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction?select=train.csv)

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

해당 모델로 어떤 유의미한 비즈니스 인사이트를 추출할 수 있는지 알아보겠습니다.
※Age(연령) — Vehicle_Damage(차 사고 경험) 비교 PDP interaction
![image](https://user-images.githubusercontent.com/70046278/112586466-33ea2980-8e3f-11eb-8622-f7ab4ad787f6.png)

Vehicle_Damage(차 사고 경험, 2=있다) / Age(연령) _ 두 변수의 상호관계(Interaction)을 보여주는 PDP(Partial Dependence Plot)
연령(Age)과 차 사고 경험(Vehicle_Damage)의 상호관계(interaction)를 비교해보니, 차 사고 경험이 있는 고객들 중 연령대 27~54세 사이가 차 보험에 긍정적일 확률이 높았습니다.
※Age(연령) — Policy_Sales_Channel(판매채널) 비교 PDP interaction
![image](https://user-images.githubusercontent.com/70046278/112586472-38164700-8e3f-11eb-9337-045e76e09379.png)

Policy_Sales_Channel(판매채널) / Age(연령) _ 두 변수의 상호관계(Interaction)을 보여주는 PDP(Partial Dependence Plot)
판매채널(Policy_Sales_Channel)과 연령(Age)을 보면, 판매채널은 채널별 편차가 커보입니다. 26, 124, 163번 채널의 연령대 27~54세 고객층이 차 보험에 긍정적일 확률이 높습니다.
데이터의 전체적인 경향을 확인해봤으니, 실제 예측된 개별 고객 데이터를 분석해보겠습니다.
PDP는 데이터 집합의 전체적 경향을 보여주고, Shap value는 개별 데이터의 경향을 확인할 수 있습니다.
아래는 ‘차 보험에 긍정적인 고객’으로 예측된 회원의 Shap Value Force Plot입니다. 예측값에 가장 영향을 많이 미친 변수는 연령(Age, 38세), 그 다음 차 사고 경험(Vehicle_Damage)이 있는 고객이었습니다.
![image](https://user-images.githubusercontent.com/70046278/112586502-449a9f80-8e3f-11eb-938d-93153e01dede.png)

한편, ‘차 보험에 부정적인 고객’으로 예측된 회원의 Shap Value Plot을 보면,
판매채널(152번), 연령(Age, 24세) 등이 예측에 중요한 영향을 미쳤습니다. 차 사고 경험이 있음에도 불구하고 연령이 어려서인지 차 보험에 대해 아직은 부정적인 것일지도 모르겠습니다.
![image](https://user-images.githubusercontent.com/70046278/112586514-482e2680-8e3f-11eb-8ef3-41ea91e26a64.png)

**< 결론 & 요약 >**

위의 데이터 분석과정을 통해 얻은 마케팅 인사이트는 크게 다음과 같습니다.
- 차 보험에 대해 긍정적인 영향을 미치는 중요한 변수는 ‘연령/차 사고 경험’ 등이며 특히 경제활동인구가 많은 27-54세 사이의 연령이 차 보험에 가장 긍정적일 확률이 높은 것으로 보아 이 연령층에 마케팅을 집중하는 방안.  
- 판매채널의 경우 채널별로 편차가 큰 것으로 보이며 26, 124, 163번 채널의 ‘핵심 연령층’에 마케팅을 집중하고, 상대적으로 낮은 반응을 보이는 판매채널에는 마케팅을 아주 적게 하거나 진행하지 않는 방법, 혹은 ‘핵심 고객층(27-54세)’을 신규로 확보하는 방향으로 마케팅 전략을 전환하는 방안

- 위 데이터에서는 판매채널에 대한 정보 및 판매정책들이 상세하게 나와있지 않기 때문에 좀 더 심층적인 마케팅 인사이트는 뽑을 수 없었습니다만, 실제 현업에서는 판매채널별/지역별 상세 정보 등을 추가로 분석해 더 뾰족한 마케팅 인사이트를 얻을 수 있을 것으로 판단됩니다.
또한, 점점 더 복잡해지고 방대해지는 고객데이터(본 포스팅에 쓰인 데이터의 양 : 381,109 rows × 12 columns)를 머신러닝 및 파이썬 분석 모듈을 활용해 빠르고 효과적으로 분석할 수 있음을 알 수 있었습니다.

**<한계점 및 보완할 사항>**
- 본 프로젝트에서는 빠른 시일 내 앙상블 모델을 비교하고, XAI 라이브러리(PDP, SHAP)로 해석하는 것에 중심을 두어 데이터 EDA, 특성공학 부분이 보완될 필요가 있음. 
- 특히, 독립특성들과 타겟의 관계를 살펴보는 Correlation Coefficients, Permutation Importance 등의 시각화를 추가할 예정.
---
## Update(21.1.18)
**1. Data EDA & Preprocessing**
- (1) 타겟과 독립특성 상관관계 시각화 추가
- (2) 불균형 클래스 문제로 undersampling 사용

**2. Modeling**
- (1) 평가지표: ROC-AUC 스코어 성능개선 (기존 best_model성능: 0.70, 수정 best_model성능: 0.85)
- (2) 과적합 문제 최소화하기 위해 GridSearchCV 사용
- (3) 머신러닝 알고리즘은 RandomForest만 사용 (boosting모델은 다음 실험에서 적용해보기로 결정) 

**3. XAI 시각화**
- (1) permutation importance 로 성능에 가장 영향을 미치는 독립특성들 순위 시각화
![image](https://user-images.githubusercontent.com/70046278/149968848-26e5fdca-d72c-419b-9ca8-70ce747a6dd6.png)

- (2) shap.force_plot 으로 검증 샘플별 예측에 가장 크게 기여한 독립특성들 확인하는 시각화
![image](https://user-images.githubusercontent.com/70046278/149969176-26b33838-a05a-42bd-a82c-55391f7fcd3e.png)

- (3) shap.summary_plot 으로 검증셋 예측에 가장 크게 기여한 독립특성들 순위 시각화 
![image](https://user-images.githubusercontent.com/70046278/149968707-49a7639b-5b68-4edc-aabd-455625240642.png)

※향후 업데이트할 계획
- best_model 저장해놓기...(다음에 사용할 때 다시 학습할 필요 없도록...) 
- XAI 시각화+인사이트 추가로 더 뽑기 (검증 샘플별 사례별로 정리해보기) 

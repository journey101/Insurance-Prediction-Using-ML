# Insurance-Prediction-Using-Machine-Learning

### 1. 프로젝트 목적 및 문제정의
1) 프로젝트 목적
 - 머신러닝 예측모델로 차 보험 가입 가능성이 높은 고객 DB 추출 
 - 선별된 고객들의 특성 중 집중할 특성을 선별하여 마케팅 전략 도출

2) 문제정의: 건강보험 가입자들 중 차 보험에 관심있을 고객들이 어떤 고객들일지 예측하는 문제.

3) 평가지표: ROC-AUC score

4) 평가지표 선정이유: 타겟이 불균형 클래스 문제를 갖고 있어 positive와 negative를 잘 분류해내는지 판단하는 ROC-AUC score를 평가지표로 사용.

### 2. 데이터 탐색 및 전처리
사용한 데이터는 미국의 한 건강보험 가입자 데이터로 데이터 특성 중 'Response(차 보험에 대한 관심 여부, 1:관심있음, 0:)' 가 최종적으로 예측하고자 하는 타겟 변수입니다.

※[데이터 출처](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction?select=train.csv)

1) 타겟(‘Response’) : 차 보험에 대한 관심 여부. (1: 관심 있음, 0: 관심 없음) 

2) 독립특성들 중 주요 변수 설명
 - Gender: 성별  / Age: 나이 / Driving_License: 운전면허 유, 무 
 - Previously_Insured: 차보험 유, 무 / Vehicle_Age: 차량 연식 / Vehicle_Damage: 차사고경험 유, 무. 
 - Annual_Premium: 연간 프리미엄. 지불 금액 / PolicySalesChannel: 판매채널코드(세부정보비공개) /. Vintage: 보험사 가입기간

3) 데이터 가공
- 1) 원본 데이터 특성 중 'Driving License(운전면허 보유여부)'데이터에서 운전면허가 있는 고객(1값)으로 1차 선별.
- 2) 'Previously_Insured(차 보험 보유 여부)' 에서 차 보험이 없는 고객 (0값) 으로 2차 선별.

### 3. 데이터 특성공학 / 모델링 

1) 범주형 데이터 인코딩: Ordinal Encoding 사용. 
- Ordinal Encoding 사용이유: 트리모델 사용할 예정으로 One-hot encoding에 비해 데이터 손실이 적고, 속도가 빠르기 때문.

2) 데이터 선별: 운전면허 있고, 차보험 없는 고객데이터만 선별.
 
3) 타겟 불균형 문제 해결: class-weight 조정한 방법과 under sampling 적용한 방법 비교

4) 모델링
- 모델1: class weight 조정 + hold-out 교차검증 + 앙상블(RandomForest)모델
> <img width="514" alt="image" src="https://user-images.githubusercontent.com/70046278/155360292-2e8b13b1-eabc-4db8-8990-47df6fbe379d.png">
> 
> 훈련 스코어가 매우 높게 나왔지만 검증 스코어가 매우 낮아 과적합 우려됨.

- 모델2: undersampling 사용 + hold-out 교차검증 + 앙상블
> <img width="477" alt="image" src="https://user-images.githubusercontent.com/70046278/155360376-9b7fcb03-e62d-4a44-95f1-c90fa60ea179.png">
> 
> 모델1보다 더 높은 검증 roc-auc 스코어가 나옴. 
> 과적합 해결을 위해 GridSearchCV 사용.

- 모델3: undersampling + GridSearchCV 교차검증 + 앙상블(RandomForest)모델
> <img width="494" alt="image" src="https://user-images.githubusercontent.com/70046278/155360479-0bfb9d05-d184-4d68-815a-21f928571402.png">
> 
> 훈련, 검증 roc-auc score 차이가 근소함. 
> 일반화 성능이 괜찮은 모델로 판단하여 모델3 채택. 학습을 추가로 진행하면 과적합 문제 더 해소할 수 있을 것.

### 4. 모델해석 
1) Permuation Importance 
: 변수별 랜덤 노이즈를 주어 무력화 시켰을 때, 모델 성능 변화를 측정하여 변수가 모델에 미치는 중요도를 측정하는 방법.
> <img width="271" alt="image" src="https://user-images.githubusercontent.com/70046278/155361169-7fa65f51-557b-4bb3-a632-81cdc6231f56.png">
> 
> 모델 예측 결과에 영향을 미치는 변수 순위 추출: 
> 1) Age : 나이
> 2) Policy_Sales_Channel: 판매채널
> 3) Region_Code: 지역코드
> 4) Annual_Premium: 연간프리미엄지불액

2) PDP Interact plot 
: 2개 특성이 모델 예측 결과에 미치는 관계 확인
> <img width="471" alt="image" src="https://user-images.githubusercontent.com/70046278/155361406-9a325889-a81c-4f65-94ff-ef0725bc2f2b.png">
> 
> 1) Age: 
> 23-54세, 경제활동인구에 많이 속하는 해당 연령대가 차보험에 관심 있을 가능성이 높음. (0.5 이상: 관심있음)
> 2) Policy_Sales_Channel: 
> 판매채널 26, 156, 163, 55, 124 채널 고객이 차보험에 관심 있을 가능성이 높음. 

3) PDP isolate plot 
: 1개 특성과 모델 예측 결과 관계 설명하는 그래프
> <img width="663" alt="image" src="https://user-images.githubusercontent.com/70046278/155361701-ea652ec2-f8cd-4ca1-b0be-6802485c240d.png">
> 
> Age: 
> 20대 중반에서 50대 중반까지는 차보험 가입 관심 가능성이 높음. 60대부터는 차보험 가입 가능성이 낮음
> 
> <img width="642" alt="image" src="https://user-images.githubusercontent.com/70046278/155361775-258c9bb8-0a76-4146-9a7e-f029a73f4764.png">
> 
> Policy_Sales_Channel: 
> 0-140 사이 채널에서는 차보험 가입 가능성이 높음. 
> 140-160 사이 채널에서는 차보험 가입 가능성이 낮음.

4) SHAP plot
: 개별 예측치에 기여한 특성 분석
(1) True Positive(예측1, 실제1) 예시 
> <img width="768" alt="image" src="https://user-images.githubusercontent.com/70046278/155362128-7e92e128-b402-40e4-acf5-7c9fc5e15749.png">
> 
> 예측결과에 기여한 특성: 
> 1) Policy_Sales_Channel: 26 / 2) Age: 54세 / 3) Vehicle_Damage: 차 사고 경험 있음(1) 

(2) True Negative(예측0, 실제0) 예시 
> <img width="651" alt="image" src="https://user-images.githubusercontent.com/70046278/155362215-03685853-c394-44d1-93e2-debf1e4a440c.png">
> 
> 예측결과에 기여한 특성
> 1) Age: 24세 / 2) Policy_Sales_Channel: 151 / 3) Vehicle_Age: 2년 

(3) False Positive(예측1, 실제0) 예시 
> <img width="702" alt="image" src="https://user-images.githubusercontent.com/70046278/155362293-e936710a-8c9c-483a-be37-e313d2ce61e1.png">
> 
> 예측결과에 기여한 특성
> 1) Age: 40세 / 2) Vehicle_Damage: 차 사고 경험 있음(1) / 3) Policy_Sales_Channel: 124

(4) False Negative(예측0, 실제1) 예시
> <img width="768" alt="image" src="https://user-images.githubusercontent.com/70046278/155362397-876db29f-f947-4c85-ad8e-9b31959469f3.png">
> 
> 예측결과에 기여한 특성
> 1) Age: 61세
> 예측 틀린 이유 추정: 나이는 61세이지만, 차 사고 경험과 판매채널 124 특성의 영향을 간과한 것이 아닐런지. 

**< 결론 & 요약 >**

위의 데이터 분석과정을 통해 얻은 마케팅 인사이트는 크게 다음과 같습니다.
- 차 보험에 대해 긍정적인 영향을 미치는 중요한 변수는 ‘나이/판매채널’ 등이며 특히 경제활동인구가 많은 20대중반에서 50대중반 사이의 연령이 차 보험에 가장 긍정적일 확률이 높은 것으로 보아 이 연령층에 마케팅을 집중하는 방안 필요. 
- 판매채널의 경우 채널별로 편차가 큰 것으로 보이며 26, 156, 163, 55, 124 채널의 ‘핵심 연령층’에 마케팅을 집중하고, 상대적으로 낮은 반응을 보이는 판매채널에는 마케팅을 아주 적게 하거나 진행하지 않는 방법, 혹은 ‘핵심 고객층(25-55연령층)’을 신규로 확보하는 방향으로 마케팅 전략을 전환하는 방안
- 위 데이터에서는 판매채널에 대한 정보 및 판매정책들이 상세하게 나와있지 않기 때문에 좀 더 심층적인 마케팅 인사이트는 뽑을 수 없었습니다만, 실제 현업에서는 판매채널별/지역별 상세 정보 등을 추가로 분석해 더 뾰족한 마케팅 인사이트를 얻을 수 있을 것으로 판단됩니다.

---
## [Update(22.2.23)](https://github.com/journey101/Insurance-Prediction-Using-ML/blob/main/Insurance-Prediction-Using-ML(update).ipynb)
**1. Data EDA & Preprocessing**
- (1) 타겟과 독립특성 상관관계 시각화 추가
- (2) 불균형 클래스 문제로 undersampling 사용

**2. Modeling**
- (1) 평가지표: ROC-AUC 스코어 일반화 성능개선 (훈련/검증 스코어 차이 줄임)
- (2) 과적합 문제 최소화하기 위해 GridSearchCV 사용 )
- (3) 모델 파일 저장

**3. XAI 시각화**

- (1) permutation importance 로 성능에 가장 영향을 미치는 독립특성들 순위 시각화
- (2) shap.force_plot 으로 검증 샘플별 예측에 가장 크게 기여한 독립특성들 확인하는 시각화
- (3) shap.summary_plot 으로 검증셋 예측에 가장 크게 기여한 독립특성들 순위 시각화 

[느낀점&배운점&향후 업데이트 계획]

우선 간략하게 느낀점은 매우 오래전부터 미뤄왔던.. update에 대한 뿌듯함과 시원함입니다. (드디어.. 해치웠..)

배운점은 크게 2가지였습니다. 
 - (1) 불균형 데이터 처리법: imblearn pipeline, undersampling 을 GridSearchCV와 같이 사용하는 법. 
 - (2) ROC-AUC-score 의미를 이해함: 모델이 0과 1을 어느정도로 명확하게 구분할 수 있는지 보여주는 지표로 높을수록 명확하게 구분함. 
   - 다만, 해당 프로젝트에서 양성 클래스의 수가 더 적고 양성 클래스를 맞추는 것을 목표로 삼고, f1-score를 평가지표로 고려했다면 더 좋았을 듯 함.
   - 물론 ROC-AUC스코어로 0과 1을 모두 잘 분류할 수 있는 모델을 지향한다면 ROC-AUC스코어를 높이고, 최적임계점을 적용하는 것도 좋을 것. 
 - (3) 언더샘플링(undersampling)과 오버샘플림(oversampling) 차이와 장단점. 
   - 언더샘플림: 과대 클래스 정보를 덜 사용하는 방법, False Positive를 줄이고 True Positive를 높여 Precision을 높이는데 좋음. 
   - 오버샘플링: 소수 클래스 정보를 더 많이 사용하는 방법, False Negative를 줄이고 True Positive를 높여 Recall을 높이는데 좋음.
   - 해당 문제에서는 언더샘플림보다는 오버샘플링을 통해 실제 True인 사람들 중 예측 결과값의 비율을 높이는 측면이 좋았을 수 있다는 생각. 왜냐하면 False Positive 로 인해 마케팅 비용이 잘못 사용되는 경우보다, True Positive를 더 많이 찾아내서 매출 상승을 극대화하는 것이 더 이윤이 될 수 있기 때문. 

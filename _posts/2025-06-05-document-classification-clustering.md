---
layout: post
title: 'Document Classification and Clustering'
date: 2025-06-05 17:17 +0900
categories: ['정보검색']
tags: ['IR', '정보검색', '강의정리']
published: true
sitemap: true
math: true
---
## Document Classification
- Classification  
    - 주어진 item이 어떤 카테고리에 속하는지 예측하는 작업.
    - 지도학습(Supervised Learning): 정답 레이블이 있는 학습 데이터로 학습
    - 예시: Naive Bayes, SVM
    - Information Retrieval에서의 활용
        - 스팸 필터링: 이메일이 스팸인지 아닌지 분류하는 작업
        - 감정 분석: 리뷰나 댓글의 감정을 긍정, 부정, 중립으로 분류하는 작업
        - 온라인 광고
    - 접근법:  
      - Probabilistic Approach: 
        - 각 클래스에 속할 확률을 계산하여 가장 높은 확률을 갖는 클래스를 선택하는 방법
        - 예시: Naive Bayes Classifier
      - Geometric Approach: 
        - 각 클래스의 특징 벡터를 정의하고, 새로운 아이템이 가장 가까운 클래스의 특징 벡터에 속하도록 분류하는 방법
        - 예시: Support Vector Machine (SVM)
    - Label 종류:
        - Binary Classification: 두 개의 클래스(예: 스팸/비스팸)
        - Multi-class Classification: 세 개 이상의 클래스(예: 뉴스 기사 분류)
        - Hierarchical Classification: 클래스가 계층 구조를 갖는 경우(예: 뉴스(최상위 클래스) → 정치, 경제, 사회(하위 클래스))

### Naive Bayes Classifier
- Naive Bayes Classifier는 확률 기반의 분류 알고리즘으로, 각 클래스에 속할 확률을 계산하여 가장 높은 확률을 갖는 클래스를 선택
- Bayes' Theorem을 기반으로 확률적 분류  

$$
Class(d) = \arg\max_{c \in C} P(c|d) = \arg\max_{c \in C} \frac{P(d|c)P(c)}{\sum_{c' \in C} P(d|c')P(c')}
$$  

    - $c$: 클래스에 해당하는 확률변수
    - $d$: 문서에 해당하는 확률변수
    - $P(d|c)$: 클래스 $c$라고 가정했을 때, 문서 $d$가 주어질 확률
    - $P(c)$: 클래스 $c$의 사전 확률
- $P(c)$의 추정  
  - class $c$를 관측할 확률은 전체 문서 수에서 class $c$에 속하는 문서의 수를 나눈 값으로 추정  
    - $P(c) = \frac{N_c}{N}$, 여기서 $N_c$는 클래스 $c$에 속하는 문서의 수, $N$은 전체 문서의 수
- $P(d|c)$의 추정  
  - 문서 $d$가 클래스 $c$에 속할 확률은 문서 $d$의 단어들을 클래스 $c$에서 관측할 확률로 추정  
$$
P(d|c) = P(w_1, w_2, \ldots, w_n | c) = P(w_1|c)P(w_2|c) \cdots P(w_n|c)
$$
    - 여기서 $w_i$는 문서 $d$의 단어들
    - 단어들이 독립적이라고 가정하여 곱셈 법칙을 적용
- **Multiple Bernoulli 모델**  
  - 문서를 단어의 유무로만 (binary vector) 표현하는 모델
  - $w_i$: 단어 $i$가 문서에 존재하는지 여부 (0 또는 1)
  - $P(w_i=1|c)$는 클래스 $c$에서 단어 $i$가 존재할 확률로 추정
  - MLE(최대 우도 추정) 방법을 사용하여 $P(w_i=1|c)$를 계산  
$$
P(w|c) = \frac{df_{w,c}}{N_c}
$$
    - $df_{w,c}$: 클래스 $c$에서 단어 $w$가 등장한 문서의 수
    - $N_c$: 클래스 $c$에 속하는 전체 문서의 수
  - $P(d|c)$는 다음과 같이 계산  
$$
P(d|c) = \prod_{w \in \mathcal{V}} P(w|c)^{\delta(w, d)} (1 - P(w|c))^{1 - \delta(w, d)}
$$
    - $\mathcal{V}$: 전체 단어 집합
    - $\delta(w, d)$: 단어 $w$가 문서 $d$에 존재하면 1, 아니면 0
    - 즉, 단어 $w$가 문서 $d$에 존재할 때는 $P(w|c)$(단어가 클래스 $c$에서 등장할 확률)을 곱하고, 존재하지 않을 때는 $(1 - P(w|c))$(단어가 클래스 $c$에서 등장하지 않을 확률)을 곱함
    - Smoothing: $P(w|c)$가 0이 되는 경우를 방지  
        - Bayesian Smoothing: $P(w|c) = \frac{df_{w,c} + \alpha}{N_c + \alpha_w + \beta_w}$
        - Laplace smoothed estimate: $P(w|c) = \frac{df_{w,c} + 1}{N_c + 1}$
        - collection smoothed estimate: $P(w|c) = \frac{df_{w,c} + \mu \frac{N_w}{N}}{N_c + \mu}$
            - $N_w$: 전체 문서에서 단어 $w$를 갖는 문서의 수

- **Multinomial 모델**  
  - 문서를 단어의 빈도로 표현하는 모델
  - $w_i$: 단어 $i$의 빈도수
  - $P(w_i|c)$는 클래스 $c$에서 단어 $i$가 등장할 확률로 추정
  - MLE 방법을 사용하여 $P(w_i|c)$를 계산  
$$
P(w|c) = \frac{tf_{w,c}}{N_c}
$$
    - $tf_{w,c}$: 클래스 $c$에서 단어 $w$가 등장한 횟수
    - $N_c$: 클래스 $c$에 속하는 전체 문서의 수
  - $P(d|c)$는 다음과 같이 계산  
$$
P(d|c) = P(|d|) \frac{|d|!}{tf_{w_1,d}!\cdots tf_{w_{\mathcal{V}},d}!} \prod_{w \in \mathcal{V}} P(w|c)^{tf_{w,d}}
$$
    - $|d|$: 문서 $d$의 단어 수
    - $tf_{w,d}$: 문서 $d$에서 단어 $w$의 빈도수
    - $\mathcal{V}$: 전체 단어 집합
    - 문서 길이 등 상수를 생략하고 비례식으로 표현하면 다음과 같다  
$$
P(d|c) \propto \prod_{w \in \mathcal{V}} P(w|c)^{tf_{w,d}}
$$
    - Smoothing: $P(w|c)$가 0이 되는 경우를 방지  
        - Bayesian Smoothing: $P(w|c) = \frac{tf_{w,c} + \alpha_w}{|c| + \sum_{w' \in \mathcal{V}} \alpha_{w'}}$
        - Laplace smoothed estimate: $P(w|c) = \frac{tf_{w,c} + 1}{|c| + |\mathcal{V}|}$
        - collection smoothed estimate: $P(w|c) = \frac{tf_{w,c} + \mu \frac{cf_w}{|C|}}{|c| + \mu}$
            - $cf_w$: 전체 문서에서 단어 $w$의 빈도수
            - $|C|$: 전체 문서에서 발생한 총 단어의 수 
            - $|c|$: 클래스 $c$에 속하는 문서의 총 단어 수

### SVM (Support Vector Machine)
- 기하학적 원리에 기반한 분류 알고리즘으로, 주어진 데이터를 분류하기 위해 두 클래스를 최대 간격으로 분리하는 초평면을 찾는 방법
  - Optimal decision boundary: 두 클래스 사이의 거리를 최대화하는 초평면
  - Support vectors: 결정 경계에 가장 가까운 데이터 포인트들
  - Margin: Support vectors와 결정 경계 사이의 거리, 최대화해야 하는 값  

- feature 함수의 집합: $f_1(), f_2(), \ldots, f_N()$
- document d는 $N$차원 벡터로 표현: $x_d = (f_1(d), f_2(d), \ldots, f_N(d))$
  - $N$: 단어 집합의 크기
- 주로 쓰이는 feature 함수: 
    - $f_w(d) = \delta(w, d)$: 단어 $w$가 문서 $d$에 존재하면 1, 아니면 0
    - $f_w(d) = tf_{w,d}$: 단어 $w$가 문서 $d$에 등장한 횟수
- **선형 분리가 가능한 경우**  
  - 목표: 마진을 최대화 하는 $w$(직선의 방향 벡터)를 찾는 것  
$$
\text{minimize } \frac{1}{2} ||w||^2
$$
    - 조건:  
$$
\begin{cases}
    w \cdot x_n \geq 1 & \text{if } Class(n) = + \\
    w \cdot x_n \leq -1 & \text{if } Class(n) = -
\end{cases}
$$
    - $x_n$: n번째 데이터 포인트
    - $Class(n)$: n번째 데이터 포인트의 클래스  

- **선형 분리가 불가능한 경우**
$$  
\text{minimize } \frac{1}{2} ||w||^2 + C \sum_{n=1}^{N} \xi_n
$$
    - $\xi_n$: slack variable, 데이터 포인트가 결정 경계에서 얼마나 벗어나는지를 나타내는 변수
    - $C$: 마진보다 오차 허용을 얼마나 중요시할지를 조절하는 하이퍼파라미터
    - 조건:  
$$
\begin{cases}
    w \cdot x_n \geq 1 - \xi_n & \text{if } Class(n) = + \\
    w \cdot x_n \leq -1 + \xi_n & \text{if } Class(n) = -
\end{cases}
$$
    - $\xi_n \geq 0$: slack variable은 항상 0 이상이어야 함  

- **SVM의 다중 클래스 분류**  
  - One-vs-All (OvA): 각 클래스에 대해 해당 클래스와 나머지 클래스를 구분하는 SVM을 학습
    - 분류 방식:  
$$ 
\text{Class}(x) = \arg \max_{c} w_c \cdot x
$$
  - One-vs-One (OvO): 각 클래스 쌍에 대해 SVM을 학습하고, 다수결 투표로 최종 클래스를 결정
    - $\frac{K(K-1)}{2}$개의 SVM을 학습해야 함, 여기서 $K$는 클래스의 개수
    - 모든 SVM의 예측 결과를 모아서 다수결 투표로 최종 클래스를 결정
    - 높은 계산복잡도  

- 평가 척도
  - **정확도 (Accuracy)**: 전체 데이터 중 올바르게 분류된 데이터의 비율
  - **정밀도 (Precision)**: 양성으로 예측한 것 중 실제 양성인 것의 비율
  - **재현율 (Recall)**: 실제 양성 중 양성으로 예측한 것의 비율
  - **F1 Score**: 정밀도와 재현율의 조화 평균
  - **ROC Curve**: True Positive Rate와 False Positive Rate의 관계를 나타내는 곡선

- feature selection  
    - document classifier는 매우 다양한 feature를 사용할 수 있지만, 모든 feature를 사용하는 것은 계산 비용이 크고, 오히려 성능을 저하시킬 수 있음
    - 유용한 feature만을 선택하여 사용하는 것이 중요

- Information Gain  
    - 각 feature가 클래스 분류에 얼마나 기여하는지를 측정하는 방법
    - feature들을 IG로 정렬하여 상위 N개의 feature만을 선택
    - multiple Bernoulli 모델의 IG:  
$$
IG(w) = H(C) - H(C|w) = -\sum_{c \in C} P(c) \log P(c) + \sum_{c \in C} P(c|w) \log P(c|w)
$$
    - 여기서 $H(C)$는 클래스 $C$의 엔트로피, $H(C|w)$는 feature $w$가 주어졌을 때의 클래스 $C$의 엔트로피
    - IG가 높을수록 해당 feature가 클래스 분류에 더 많은 정보를 제공함

- Spam 탐지  
  - spam: 나쁜 의도로 작성된 메시지, 광고, 피싱 등
  - link spam: link관련 score를 인위적으로 높이기 위한 행위
  - term spam: 특정 키워드를 과도하게 사용하여 검색 결과를 조작하는 행위
  - 유용한 feature: 단어 단위(unigram), 표현 형식(보이지 않는 문자, HTML 태그 등), 오타, IP 주소 등

- Sentiment Classification  
  - 감정 분석: 텍스트에서 감정을 추출하는 작업
  - 긍정, 부정, 중립 등으로 분류
  - 의견의 강도도 중요
  - 유용한 feature: 단어 단위(unigram), PoS 태깅, 감정 사전 등
  - 문맥(Context)도 중요: 문장의 의미는 단어의 순서와 문맥에 따라 달라질 수 있음
  - unigram feature를 사용한 SVM이 수동 규칙 집합보다 우수한 성능

## Document Clustering
- Clustering  
    - 주어진 item을 비슷한 item끼리 묶는 작업.
    - 비지도학습(Unsupervised Learning): 정답 레이블이 없는 학습 데이터로 학습, 유사도를 기반으로 데이터 포인트를 그룹화
    - Feature를 어떻게 정의할지, similarity를 어떻게 정의할지에 따라 결과가 달라짐
    - 일반적 알고리즘  
        - 아이템의 feature vector 정의
        - similarity measure 정의
        - 좋은 클러스터의 기준 정의
        - 반복적으로 클러스터링
    - 예시: K-means, Hierarchical Clustering

### Hierarchical Clustering
- 계층 구조를 만드는 클러스터링 방법
  - 가장 위 계층: 전체 데이터
  - 가장 아래 계층: 각 데이터 포인트
- 두가지 접근법
  - Agglomerative (Bottom-Up): 각 데이터 포인트를 개별 클러스터로 시작하여, 가장 가까운 클러스터들을 합쳐 나가는 방식
    1. 각 데이터 포인트를 개별 클러스터로 시작
    2. 가장 가까운 두 클러스터를 찾아 합침
    3. distance matrix를 업데이트
    4. 모든 데이터 포인트가 하나의 클러스터로 합쳐질 때까지 반복
  - Divisive (Top-Down): 전체 데이터를 하나의 클러스터로 시작하여, 가장 먼 데이터 포인트를 기준으로 클러스터를 분할해 나가는 방식
- Dendrogram(트리 구조)를 사용하여 클러스터링 결과를 시각화
- 두 클러스터 간 거리 측정법
  - Single Linkage: 두 클러스터 간 가장 가까운 데이터 포인트 간의 거리  
$$
COST(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)
$$
  - Complete Linkage: 두 클러스터 간 가장 먼 데이터 포인트 간의 거리  
$$
COST(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)
$$
  - Average Linkage: 두 클러스터 간 모든 데이터 포인트 간의 평균 거리  
$$
COST(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)
$$
  - Average Group Linkage: 두 클러스터 중심점 간의 거리  
$$
COST(C_i, C_j) = d(\mu_{C_i}, \mu_{C_j})
$$
    - $\mu_{C_i}$: 클러스터 $C_i$의 중심점
  - Ward's method: 클러스터 합병 시 분산의 증가량을 최소화하는 방법  
$$
COST(C_i, C_j) = \sum_{k \neq i,j} \sum_{X \in C_k} (X - \mu_{C_k})^2 + \sum_{X \in C_i \cup C_j} (X - \mu_{C_i \cup C_j})^2
$$
    - 여기서 $X$는 데이터 포인트, $\mu_{C_k}$는 클러스터 $C_k$의 중심점
    - 앞 항은 병합되지 않은 나머지 클러스터들의 분산 총합, 뒤 항은 병합된 클러스터의 내부 분산

### K-means Clustering
- K-means는 주어진 데이터 포인트를 K개의 클러스터로 나누는 알고리즘
- 각 클러스터는 중심점(centroid)으로 표현되고, 각 데이터 포인트는 가장 가까운 중심점에 할당됨
- 알고리즘 단계
  1. K개의 중심점을 무작위로 초기화
  2. 각 데이터 포인트를 가장 가까운 중심점에 할당
  3. 각 클러스터의 중심점을 재계산
  4. 중심점이 더 이상 변화하지 않을 때까지 2-3단계를 반복
- 최적화 수식  
$$
\text{minimize } \sum_{k=1}^{K} \sum_{x \in C_k} ||x - \mu_k||^2
$$
    - $C_k$: 클러스터 $k$에 속하는 데이터 포인트 집합
    - $\mu_k$: 클러스터 $k$의 중심점

### k-NN Clustering
- 위 두 방법과 달리, 하나의 item마다 하나의 클러스터를 형성(중첩 가능)
- 하나의 item과 그 최근접 k개의 item을 클러스터로 정의
- 따라서 데이터 포인트의 개수만큼 클러스터가 생성됨
- **k 값 정하기**
  - Kmeans, k-NN 모두 k값은 하이퍼파라미터로, 적절한 k값을 찾는 것이 중요
  - adaptive k: 데이터의 밀도에 따라 k값을 조정하는 방법

### Evaluation of Clustering
- 클러스터링은 비지도 학습이므로, 평가가 어려움
- 정답 레이블이 있다면 동일하게 Precision, Recall, F1 Score 등을 사용할 수 있지만, 일반적으로는 다음과 같은 방법을 사용
- Cluster Precision  
$$
\text{Cluster Precision} = \frac{1}{N} \sum_{i=1}^{K} |\text{MaxClass}(C_i)|
$$
  - $N$: 전체 데이터 포인트 수
  - $K$: 클러스터의 개수
  - $C_i$: i번째 클러스터
  - $\text{MaxClass}(C_i)$: i번째 클러스터에서 가장 많이 등장하는 클래스의 개수
  - 클러스터마다 클래스가 얼마나 잘 나눠져 있는지를 평가

### Clustering and IR
- Document Clustering: 문서들을 유사한 주제나 내용을 가진 그룹으로 묶는 작업
- 가정: 가까운 문서들은 비슷한 query에 대해 검색됨
- 검색 모델
  - Clustering 기반 검색:  
$$
P(Q|C_j) = \prod_{i=1}^{N} P(q_i|C_j)
$$
    - $Q$: query
    - $C_j$: j번째 클러스터
    - $q_i$: query의 i번째 단어
    - 개별 document를 바로 ranking하는 것이 아니라, 클러스터를 ranking한 후에 document를 ranking하는 방식
  - k-NN 기반 Smoothing:  
$$
P(w|D) = (1- \lambda - \delta) \frac{f_{w,D}}{|D|} + \delta \sum_{C_j} \frac{f_{w,C_j}}{|C_j|} P(D|C_j) + \lambda \frac{f_{w,Coll}}{|Coll|}
$$
    - $D$: document
    - $C_j$: j번째 클러스터
    - $f_{w,D}$: document D에서 단어 w의 빈도수
    - $|D|$: document D의 총 단어 수
    - $f_{w,C_j}$: 클러스터 C_j에서 단어 w의 빈도수
    - $|C_j|$: 클러스터 C_j의 총 단어 수
    - $P(D|C_j)$: document D가 클러스터 C_j에 속할 확률
    - $f_{w,Coll}$: 전체 문서 집합에서 단어 w의 빈도수
    - $|Coll|$: 전체 문서 집합의 총 단어 수
    - $\lambda$: smoothing parameter, 0과 1 사이의 값
    - $\delta$: k-NN에서 k개의 최근접 이웃을 고려하는 비율
    - document D에서 단어 w의 확률을 계산할 때, document D 자체의 정보, 클러스터 C_j의 정보, 전체 문서 집합의 정보를 모두 고려

---
해당 포스트는 서울대학교 산업공학과 박종헌 교수님의 데이터관리와 분석 25-1학기 강의를 정리한 내용입니다.  
---
layout: post
title: 'Evaluating Search Engines'
date: 2025-06-07 01:08 +0900
categories: ['정보검색']
tags: ['IR', '정보검색', '강의정리']
published: true
sitemap: true
math: true
---
## Evaluating Search Engines
- 검색엔진의 성능을 평가하는 것은 정보 검색 시스템의 품질을 보장하는 데 필수적
- 효율성(efficiency)와 효과성(effectiveness)과 cost는 서로 trade-off 관계에 있음

### 평가용 Documents
- 검색엔진의 성능을 평가하기 위해서는 평가용 문서가 필요
- document, query, relevance로 구성
- 대표적 평가용 문서로 CACM, AP, GOV2, TREC 등이 있음
- TREC은 Text REtrieval Conference의 약자로, 정보 검색 시스템의 성능을 평가하기 위한 대회
  - 각 topic은 다음과 같이 구성
    - `<num>`: 번호
    - `<title>`: 제목
    - `<desc>`: 사용자의 정보 요구 설명
    - `<narr>`: 관련 문서를 정의하는 기준 설명

### Relevance Judgments
- 관련선 판단 데이터는 수집과 비용이 많이 드는 작업
  - 판단자는 누구인지, 얼마나 많은 문서에 대해 판단하는지, 어떤 기준으로 판단하는지 등이 중요
  - 판단자 간의 일관성도 중요
- TREC에서는
  - binary relevance: 문서가 관련 있는지 없는지
  - narrative를 추가해 판단자간 일관성을 높임
  - pooling 기법을 사용해 다양한 판단자를 통해 문서의 관련성을 평가

### Query log
- 검색엔진의 성능을 평가하기 위해서는 사용자 쿼리 로그가 필요
- 구성 요소:
  - ID(user or session) : 사용자 또는 세션의 고유 식별자
  - query term : 사용자가 입력한 검색어
  - timestamp : 쿼리가 입력된 시간
  - URL list : 결과로 반환된 URL 목록(순위, 클릭 여부 등 포함)
- 클릭했다고 해서 해당 URL이 관련성이 있다고 판단할 수는 없지만, 간접적 으로 관련성을 추정할 수 있음
  - 문서 끼리의 preference를 나타내는 것으로는 볼 수 있음(여러 문서 중 하나를 클릭했다는 것은 다른 문서보다 더 관련성이 있다고 판단했음을 의미)

### Evaluation Metrics  

#### Effectiveness
- 검색엔진의 효과성을 평가하기 위한 지표
- Recall:
  - 전체 관련 문서 중 검색된 관련 문서의 비율  
$$
\text{Recall} = \frac{|R \cap N|}{|R|}
$$
    - R: 전체 관련 문서 집합, N: 검색된 문서 집합
- Precision:
  - 검색된 문서 중 관련 문서의 비율  
$$
\text{Precision} = \frac{|R \cap N|}{|N|}
$$
    - R: 전체 관련 문서 집합, N: 검색된 문서 집합
- F-measure:
  - Precision과 Recall의 조화 평균  
$$
\text{F-measure} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$
    - Precision과 Recall의 trade-off를 고려한 지표
  - 일반화 형태  
$$
\text{F}_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$
    - $\beta$는 Precision과 Recall의 중요도를 조절하는 파라미터

- **AP(average precision)**
  - Precision의 평균값으로, 관련 문서가 검색된 각 위치의 Precision을 계산하여 평균을 구함  
  - 예시:
    - 검색된 문서가 10개이고, 전체 관련 문서가 5개, 검색된 관련 문서가 3개인 경우
    - Ranking #1: [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
      - Precision = [1, 0.5, 0.67, 0.5, 0.6, 0.5, 0.43, 0.38, 0.33, 0.3]
      - AP = (1 + 0.5 + 0.67) / 3 = 0.723
    - Ranking #2: [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]  
      - Precision = [1, 0.5, 0.33, 0.5, 0.4, 0.33, 0.43, 0.38, 0.33, 0.3]
      - AP = (1 + 0.5 + 0.33) / 3 = 0.611

- **MAP(mean average precision)**
  - 여러 쿼리에 대한 AP의 평균값
  - 사용자는 가능한 한 많은 문서를 찾기 윈한다고 가정
  - recall-precision graph를 통해 ranking을 시각적으로 요약 가능
    - interpolation: graph의 평균을 구하기 위해 recall level $R$에서의 precision $P$를 구함  
$$
P(R) = \max\{P': R' \geq R \cap (R', P') \in S\}
$$
    - S: precision-recall graph의 점들
    - 주어진 recall $R$에서 precision $P$는 $R$ 이상의 recall을 가지는 점들 중에서 최대 precision을 선택
    - Recall 0에서의 precision도 정의 가능

- 상위 문서의 우대
  - 사용자들은 상위 문서만을 검토하는 경향이 있음
  - 따라서 단순 Recall보단, 얼마나 relevant한 문서가 상위에 위치하는지가 중요
  - 주요 측정 지표:  
    - Precision at k (P@k):
      - 상위 k개의 문서 중 관련 문서의 비율
      - 예시: P@10은 상위 10개 문서 중 관련 문서의 비율
    - Reciporal Rank (RR):
      - 관련 문서가 처음 등장하는 위치의 역수
      - 예시: 관련 문서가 3번째에 등장하면 RR = 1/3
    - Mean Reciprocal Rank (MRR):
      - 여러 쿼리에 대한 RR의 평균값
      - 특히 문서가 하나만 관련된 경우에 유용
      - 예시: MRR = (1/3 + 1/5 + 1/2) / 3

- **DCG (Discounted Cumulative Gain)**
  - 검색 결과의 순위에 따라 관련 문서의 가치를 다르게 평가
  - 순위가 높을수록 더 높은 가중치를 부여
  - DCG는 다음과 같이 계산  
$$
\text{DCG}_k = rel_1 + \sum_{i=2}^{k} \frac{rel_i}{\log_2(i)}
$$
    - $rel_i$: i번째 문서의 관련성 점수
    - k: 상위 k개의 문서
    - 가중치: $\frac{1}{\log_2(i)}$로, 순위가 높을수록 가중치가 커짐

- **NDCG (Normalized DCG)**
    - 어떤 쿼리는 relevant 많고, 어떤 쿼리는 relevant가 적을 수 있음
    - 공정한 기준으로 비교하기 위해 정규화된 DCG 사용  
$$
\text{NDCG}_k = \frac{\text{DCG}_k}{\text{IDCG}_k}
$$
    - IDCG는 이상적인 순위에서의 DCG로, 관련 문서가 최상위에 위치한 경우의 DCG
    - NDCG는 0과 1 사이의 값을 가지며, 1에 가까울수록 이상적인 순위를 의미

- **BPREF (Binary Preference)**
  - 관련 문서외 비관련 문서가 불균형할 때, 균형을 맞춰 평가하기 위한 지표
  - $R$개의 관련 문서가 있을 때, 상위 $R$개의 검색 결과 중 비관련 문서만 평가  
$$
\text{BPREF} = \frac{1}{R} \sum_{d_r} (1 - \frac{N_{d_r}}{R})
$$
    - $d_r$: 관련 문서, $N_{d_r}$: 관련 문서 $d_r$보다 상위에 있는 비관련 문서의 수(최대 $R$개의 document 중)  

#### Efficiency
- 검색엔진의 효율성을 평가하기 위한 지표
- Elapsed indexing time: 색인 생성에 걸린 시간
- Indexing processor time: 색인 생성에 걸린 CPU 시간
- Query throughput: 초당 처리할 수 있는 쿼리 수
- Query latency: 쿼리 처리에 걸린 시간
- Indexing temporary space: 색인 생성에 사용된 임시 공간
- Index size: 색인 파일의 크기

### Online Test
- 실제 사용자 트래픽 일부에 실험용 검색엔진을 적용해 성능을 평가(A/B 테스트)
- **Pros**
  - 실제 사용자 데이터를 기반으로 평가 가능
  - 검색엔진의 실제 사용 환경에서 성능을 평가할 수 있음
- **Cons**
  - 데이터에 노이즈가 많아, 정확한 평가가 어려울 수 있음
  - 사용자 경험을 해치는 위험이 있음
  - 따라서 일부(1~5%) 트래픽에 대해서만 적용
- 통계적 검정 필요

---
해당 포스트는 서울대학교 산업공학과 박종헌 교수님의 데이터관리와 분석 25-1학기 강의를 정리한 내용입니다.  
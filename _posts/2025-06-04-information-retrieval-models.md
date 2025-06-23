---
layout: post
title: 'Information Retrieval Models'
date: 2025-06-04 21:59 +0900
categories: ['정보검색']
tags: ['IR', '정보검색', '강의정리', '자연어처리']
published: true
sitemap: true
math: true
---
## Information Retrieval(IR) Models
Information Retrieval(IR) 모델은 정보 검색 시스템에서 문서와 쿼리 간의 관련성을 평가하고, 검색 결과를 랭킹하는 데 사용되는 다양한 방법론을 의미한다.
- Relevance  
    문서와 쿼리 간의 관련성 정의는 사람마다 다를 수 있다.
- 따라서 검색 시스템을 수리적으로 표현하는 틀이 필요하다.
- 많은 Ranking Algorithm의 기초
- 쿼리와 문서의 주제 관련성, 품질 등을 Ranking function으로 계산해 Document score를 계산함

### 초기 모델
#### Boolean Model
문서와 쿼리를 이진 값으로 표현하여, 문서가 쿼리와 일치하는지 여부를 판단하는 모델.
  - 문서와 쿼리를 단어의 집합으로 표현
  - 문서가 쿼리와 일치하면 True, 일치하지 않으면 False로 표현
  - AND, OR, NOT 연산 등 boolean 연산자를 사용하여 쿼리를 구성
    - 예: "`이순신 AND 장군 OR 제독 NOT 거북선`": 이순신과 관련된 문서 중 장군 또는 제독이 포함되지만 거북선은 제외된 문서를 검색
    - proximity 연산자도 사용됨(예: `이순신 장군~2`): 이순신과 장군이 2단어 이내에 있는 문서를 검색
  - **Pros**
    - 예측 가능하고 설명이 용이
    - 다양한 feature를 사용하여 쿼리를 구성할 수 있음
    - 빠른 검색 속도
  - **Cons**  
    - 사용자에 따라 검색 효율성 차이
    - 쿼리가 단순하면 너무 많은 결과, 복잡하면 너무 적은 결과

#### Vector Space Model
문서와 쿼리를 벡터로 표현하여, 두 벡터간의 유사도를 활용하여 문서의 관련성을 평가하는 모델.
  - document와 query가 term weight의 벡터로 표현됨
  - document collection은 term weight의 행렬로 표현됨  

$$
Q = (q_1, q_2, \ldots, q_t), \quad D_i = (d_{i1}, d_{i2}, \ldots, d_{it})
$$  
$$
D = \begin{pmatrix}
d_{11} & d_{12} & \ldots & d_{1t} \\
d_{21} & d_{22} & \ldots & d_{2t} \\
\vdots & \vdots & \ddots & \vdots \\
d_{n1} & d_{n2} & \ldots & d_{nt}
\end{pmatrix}
$$

  - $Q$: 쿼리 벡터($q_i$: $i$번째 term의 쿼리 포함 여부 혹은 빈도수 등의 가중치)
  - $D_i$: $i$번째 문서 벡터($d_{ij}$: $i$번째 문서에 $j$번째 term이 포함된 정도)
  - $D$: 문서 행렬($d_{ij}$: $i$번째 문서에 $j$번째 term이 포함된 정도)
  - 문서와 쿼리 간의 유사도는 코사인 유사도(Cosine Similarity)를 사용하여 계산됨
  - **Pros**
    - 문서와 쿼리 간의 유사도를 수치적으로 표현할 수 있어, 검색 결과를 랭킹할 수 있음
    - 어떠한 유사도 또는 term weight를 사용하더라도 적용 가능
  - **Cons**
    - term의 독립성을 가정

- **Term weight (TF-IDF)**
  - Term Frequency (TF): 특정 term이 문서 내에서 얼마나 자주 등장하는지를 나타내는 가중치(단어가 문서 내 얼마나 자주 등장하는지)  
$$
\text{tf}_{ik} = \frac{f_{ik}}{\sum_{j=1}^{t} f_{ij}}
$$
    - $f_{ik}$: $i$번째 문서에서 $k$번째 term의 빈도수
    - $\sum_{j=1}^{t} f_{ij}$: $i$번째 문서의 전체 term 빈도수 합
  - Inverse Document Frequency (IDF): 특정 term이 전체 문서에서 얼마나 희귀한지를 나타내는 가중치(너무 많은 문서에 등장하는 term은 중요도가 낮다고 가정)  
$$
\text{idf}_k = \log\left(\frac{N}{n_k}\right)
$$
    - $N$: 전체 문서 수
    - $n_k$: $k$번째 term이 포함된 문서 수
  - TF-IDF:  
$$
\text{tf-idf}_{ik} = \frac{(\log(f_{ik} + 1)) \cdot \log\left(\frac{N}{n_k}\right)}{\sqrt{\sum_{k=1}^{t} (\log(f_{ik} + 1) \cdot \log\left(\frac{N}{n_k}\right))^2}}
$$
    - $f_{ik}$: $i$번째 문서에서 $k$번째 term의 빈도수
    - $\sum_{j=1}^{t} f_{ij}$: $i$번째 문서의 전체 term 빈도수 합

### 확률 모델
#### 검색 문제와 분류 문제
- $P(R|D)$: 문서 $D$가 주어졌을 때, 문서가 관련성이 있을 확률
- $P(NR|D)$: 문서 $D$가 주어졌을 때, 문서가 관련성이 없을 확률
- 검색 문제를 문서 $D$가 주어졌을 때, 관련 문서인지 아닌지 판단하는 이진 분류 문제로 볼 수 있다.

- **Bayes Classifier**
  - Bayes Rule:  
$$
P(R|D) = \frac{P(D|R) \cdot P(R)}{P(D)}
$$
  - 의사결정 기준:  
    - $P(R|D) > P(NR|D)$이면 문서 $D$는 관련성이 있다고 판단
    - 우변의 확률을 정리해서 다음과 같이 표현할 수 있다.  
$$
P(R|D) = \frac{P(D|R) \cdot P(R)}{P(D)} \gt P(NR|D) = \frac{P(D|NR) \cdot P(NR)}{P(D)}
$$  
$$
\frac{P(D|R)}{P(D|NR)} \gt \frac{P(NR)}{P(R)}
$$
  - 좌항: 문서가 관련문서일때 생성될 가능성 대비 비관련일 가능성
  - 우항: 관련 문서와 비관련 문서의 사전 확률(Prior Probability)
  - $P(D|R)$ 추정  
    - Term independence Assumption: 문서 내의 단어들이 서로 독립적이라고 가정
    - 문서 내의 단어들의 확률을 곱하여 계산
    - Document $D = (d_1, d_2, \ldots, d_t)$라고 할 때, $P(D|R) = \prod_{i=1}^{t} P(d_i|R)$
      - $P(d_i|R)$: 문서 $D$가 관련 문서일 때, 단어 $d_i$가 등장할 확률
  - 어떤 문서를 binary vector로 보고(term의 존재 여부로 표현), 확률 비율로 비교  
$$
\frac{P(D|R)}{P(D|NR)} = \prod_{i:d_i = 1} \frac{p_i}{s_i} \cdot \prod_{i:d_i = 0} \frac{1 - p_i}{1 - s_i} = \prod_{i:d_i = 1} \frac{p_i(1 - s_i)}{s_i(1 - p_i)} \prod_i \frac{1 - p_i}{1 - s_i}
$$
    - $p_i$: 관련 문서에서 단어 $d_i$가 등장할 확률
    - $s_i$: 비관련 문서에서 단어 $d_i$가 등장할 확률
  - **BIM Scoring Function**  
    - 위 식을 로그를 취해 점수 형태로 표현  
$$
\sum_{i:d_i = 1} \log\left(\frac{p_i(1 - s_i)}{s_i(1 - p_i)}\right)
$$  
    - 현실 세계에선 관련 문서와 비관련 문서의 확률을 정확히 알 수 없으므로, $p_i=0.5$, $s_i$는 전체 코퍼스에서 추정(관련 문서 수가 전체 문서 수 대비 매우 작다고 추정하고, 전체 문서에서 단어 $d_i$가 등장하는 비율로 추정)  
$$
\log\left(\frac{0.5(1 - s_i)}{s_i(1 - 0.5)}\right) = \log\left(\frac{1 - s_i}{s_i}\right) \approx \log\left(\frac{N - n_i}{n_i}\right)
$$
    - IDF와 거의 유사한 형태로 추정
  - **Contingency Table**  
    - 관련 문서와 비관련 문서에서 term occurrece 정보가 있는 경우, contingency table을 사용  
    
    |         | Relevant | Non-Relevant  | Total   |
    | ------- | -------- | ------------- | ------- |
    | $d_i=1$ | $r_i$    | $n_i-r_i$     | $n_i$   |
    | $d_i=0$ | $R-r_i$  | $N-n_i-R+r_i$ | $N-n_i$ |
    | Total   | $R$      | $N-R$         | $N$     |

    - $p_i = \frac{r_i}{R}$, $s_i = \frac{n_i - r_i}{N - R}$로 표현 가능
      - $\log0$ 문제가 있음
    - 개선된 추정치:  
$$
p_i = \frac{r_i + 0.5}{R + 1}, \quad s_i = \frac{n_i - r_i + 0.5}{N - R + 1}
$$
      - $0.5$는 smoothing parameter로, 0으로 나누는 문제를 방지하기 위해 사용
      - $1$은 전체 문서 수를 고려하여 smoothing을 적용하기 위해 사용
    - 최종 BIM Scoring Function:  
$$
\sum_{i:d_i = q_i = 1} \log \left(\frac{r_i + 0.5}{R - r_i + 0.5} \cdot \frac{N - R - n_i + r_i + 0.5}{n_i - r_i + 0.5}\right)
$$
  - 한계: 주어진 쿼리에 대한 $r_i$는 대부분의 경우에 알 수 없고, 이 경우 scoring 함수는 단순 IDF 가중치로 작동

#### BM25  
BM25는 BIM 기반의 ranking 함수, 문서와 쿼리 의 TF-IDF를 기반으로 하여, 문서의 길이와 쿼리의 길이를 고려한 확률적 모델이다.
$$
\sum_{i \in Q} \log \left(\frac{r_i + 0.5}{R - r_i + 0.5} \cdot \frac{N - R - n_i + r_i + 0.5}{n_i - r_i + 0.5}\right) \cdot \frac{(k_1 + 1) \cdot f_i}{K + f_i} \cdot \frac{(k_2 + 1) \cdot qf_i}{k_2 + qf_i}
$$  
$$
K = k_1 \cdot \left(1 - b + b \cdot \frac{dl}{avdl}\right)
$$
- $f_i$: 문서에서 $i$번째 term의 빈도수
- $qf_i$: 쿼리에서 $i$번째 term의 빈도수
- $k_1, k_2, K$: 조정 파라미터
  - $k_1$: 문서 길이에 대한 조정 파라미터(일반적으로 $k_1 \in [1.2, 2.0]$)
  - $k_2$: 쿼리 길이에 대한 조정 파라미터(일반적으로 $k_2 \in [0, 100]$)
- $dl$: 문서의 길이
- $avdl$: 전체 문서의 평균 길이
- $b$: 문서 길이에 대한 조정 파라미터(일반적으로 $b \in [0.5, 0.9]$)
- relevance 정보가 없는 경우 $r_i$와 $R$는 0으로 설정

#### Language Model
- **Unigram Language Model**
  - 각 term이 독립적으로 발생한다고 가정
  - 단어 선택은 고정된 확률 분포에 따라 이루어진다고 가정
  - 이전 단어의 영향을 고려하지 않음
- **n-gram Language Model**
  - 단어의 순서를 고려하여, n개의 연속된 단어를 하나의 단위로 취급
  - 예: bigram은 두 단어의 연속, trigram은 세 단어의 연속
- **Topic-based Language Model**
  - 특정 topic은 관련된 단어들의 공동 분포를 가지고 있음
    - 예를 들어, "축구"라는 topic은 "골", "공격", "수비" 등의 단어가 자주 등장함
  - 이를 multinomial distribution로 모델링  

$$
P(\mathbf{x_1} = x_1, \mathbf{x_2} = x_2, \ldots, \mathbf{x_n} = x_n) = \begin{cases}
    \frac{n!}{x_1! \cdot x_2! \cdots x_k!} \cdot p_1^{x_1} \cdot p_2^{x_2} \cdots p_k^{x_k} & \text{if } \sum_{i=1}^{k} x_i = n \\
    0 & \text{otherwise}
\end{cases}
$$  
  - $x_i$: $i$번째 단어의 등장 횟수
  - $p_i$: $i$번째 단어가 topic에서 등장할 확률
  - $n$: 문서 내의 단어의 총 등장 횟수
  - $k$: topic 내의 단어의 종류 수
  - 각 topic에 대해 위 공식으로 확률을 계산해 제일 높은 확률을 가지는 topic을 선택

- **Language Model for Information Retrieval**
  - $P(D|Q)$  
    쿼리 언어 모델이 있을 때, 문서가 생성될 확률 평가
  - $P(Q|D)$  
    문서 언어 모델이 있을 때, 쿼리가 생성될 확률 평가
  - 두 모델간 유사도 평가
- **Query Likelihood Model**
    - 우리가 관심 있는 것
      - $P(D|Q)$: 쿼리 $Q$가 주어졌을 때, 문서 $D$가 생성될 확률
      - 베이즈 정리를 이용하여 다음과 같이 표현  
$$
P(D|Q) = \frac{P(Q|D) \cdot P(D)}{P(Q)}
$$
    - 한 쿼리 $Q$에 대해, $P(Q), P(D)$는 고정된 값이므로, $P(Q|D)$를 최대화하는 문서 $D$를 찾는 것이 목표
    - 기본 수식:  
$$
P(Q|D) = \prod_{i=1}^{n} P(q_i|D)
$$
      - $q_i$: 쿼리 $Q$의 $i$번째 단어
      - $P(q_i|D)$: 문서 $D$에서 단어 $q_i$가 등장할 확률
    - MLE (Maximum Likelihood Estimation) 추정  
$$
P(q_i|D) = \frac{f_{q_i, D}}{|D|}
$$ 
      - $f_{q_i, D}$: 문서 $D$에서 단어 $q_i$의 빈도수
      - $|D|$: 문서 $D$의 총 단어 수
      - 문제점: $P(q_i|D) = 0$인 경우 전체 확률이 0이 되어버림
    - **Smoothing** 기법을 사용하여 문제 해결
      - Jelinek-Mercer Smoothing:  
$$
P(q_i|D) = (1 - \lambda) \cdot \frac{f_{q_i, D}}{|D|} + \lambda \cdot \frac{c_{q_i}}{|C|}
$$
        - $\lambda$: smoothing parameter (0과 1 사이의 값)
        - $c_{q_i}$: 전체 코퍼스에서 단어 $q_i$의 빈도수
        - $|C|$: 전체 코퍼스의 총 단어 수
        - 문서에 있는 정보를 많이 반영하면서, 없는 단어에 대해서는 전체 코퍼스에서의 빈도수를 반영
        - TF에 비례하고(문서 내에서의 빈도수 $f_{q_i, D}$), DF에 반비례($\frac{c_{q_i}}{|C|}$)하는 형태로 나타남
$$
P(q_i|D) = \frac{f_{q_i, D} + \mu \cdot \frac{c_{q_i}}{|C|}}{|D| + \mu}
$$
        - $\mu$: smoothing parameter (문서의 길이에 따라 조정)
        - 문서의 길이에 따라 smoothing을 적용하여, 짧은 문서에서도 효과적으로 작동
    - ranking을 위해 score로 변환할 때, 로그를 취하면 수치 계산이 더 안정적

#### Relevance Models
- **Relevance Model**
    - query를 relevance language model $R$ 에서 생성된 것으로 추정
    - query와 relevant document는 같은 language model $R$을 공유한다고 가정
    - 목적:  
      - $P(D|R)$: relevance model $R$이 주어졌을 때, 문서 $D$를 생성할 확률
      - $R$은 쿼리로부터 추정
      - document likelihood: 쿼리에 대한 relevance document의 표본이 주어졌을 때, $R$의 확률들을 추정  
$$
P(R|Q) = \frac{P(Q|R) \cdot P(R)}{P(Q)}
$$

- **KL Divergence**
    - KL Divergence: 두 확률 분포 $P$와 $Q$ 사이의 차이를 측정하는 방법  
$$
{KL}(P || Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)
$$
    - $P(w|R)$: 쿼리로부터 도출한 relevance model $R$(true distribution)
      - query $Q$에서 word $w$의 frequency 기반으로 계산  
$$
P(w|R) = \frac{f_{w, Q}}{|Q|}
$$
      - $f_{w, Q}$: 쿼리 $Q$에서 단어 $w$의 빈도수
      - $|Q|$: 쿼리 $Q$의 총 단어 수
    - $P(w|D)$: 문서 $D$의 언어 모델(approximate distribution)
    - KL Divergence를 사용하여 두 분포 간의 차이를 측정하고 음수화해서 유사도를 점수화  
$$
-{KL}(P(w|R) || P(w|D)) = -\sum_{w} P(w|R) \log\left(\frac{P(w|R)}{P(w|D)}\right) = -\sum_{w} P(w|R) \log(P(w|R)) + \sum_{w} P(w|R) \log(P(w|D))
$$  
$-\sum_{w} P(w|R) \log(P(w|R))$는 상수이므로, 최종적으로 다음과 같은 점수로 표현할 수 있다.
$$
\sum_{w} P(w|R) \log(P(w|D)) = \sum_{w} \frac{f_{w, Q}}{|Q|} \log P(w|D)
$$
    - $P(w|R)$: relevance model에서 단어 $w$의 확률
    - $P(w|D)$: 문서 $D$에서 단어 $w$의 확률
    - 이 점수는 문서 $D$가 relevance model $R$에 얼마나 잘 맞는지를 나타냄
    - 이는 Query Likelihood Model의 특수한 형태($P(Q|D) = \prod_{i=1}^{n} P(q_i|D)$)로 볼 수도 있음

- Relevance Feedback
    - 사용자가 직접 relevant document를 선택
    - 선택된 document를 기반으로 query 확장 및 reranking
    - 학습 데이터가 매우 적은 머신러닝 알고리즘

- Pseudo Relevance Feedback
    - 사용자가 직접 document를 선택하지 않고, 검색 결과에서 상위 $k$개의 document를 relevant document로 간주
    - 선택된 document를 기반으로 query 확장 및 reranking
    - 사용자의 피드백 없이도 relevance model을 개선
    - 알고리즘:
      - 초기 query로 검색을 수행하여 상위 $k$개의 document를 선택
      - 선택된 document를 기반으로  새로운 relevance model $R$을 추정
      - 새로운 relevance model $R$을 다시 reranking
      - 상위 문서에서 자주 등장하는 단어를 쿼리에 추가
    - 예:  
      - 초기 쿼리: "이순신"
      - 상위 $k$개의 document에서 "장군", "전쟁", "해전" 등의 단어가 자주 등장
      - 확장된 쿼리: "이순신 장군 전쟁 해전"

- Relevance Model 추정
  - 목표: 단어 $w$가 relevance model $R$에서 등장할 확률을 추정
  - 가정: 단어 $w$를 추출할 확률은 이전에 추출된 $n$개의 쿼리 단어에 따라 다르다고 가정  
$$
P(w | q_1, q_2, \ldots, q_n) \approx P(w | R)
$$
  - Joint 확률:  
$$
P(w, q_1, q_2, \ldots, q_n) = \sum_{D \in C} P(D) P(w, q_1, q_2, \ldots, q_n | D)
$$
  - 조건부 독립성 가정:  
$$
P(w, q_1, q_2, \ldots, q_n | D) = P(w | D) \prod_{i=1}^{n} P(q_i | D)
$$
  - 최종적으로, relevance model $R$에서 단어 $w$의 확률은 다음과 같이 표현됨  
$$
P(w, q_1, q_2, \ldots, q_n) = \sum_{D \in C} P(D) P(w | D) \prod_{i=1}^{n} P(q_i | D)
$$
    - $P(D)$(문서 $D$가 선택될 확률) 일반적으로 uniform distribution으로 가정하므로 무시
    - 위 식은 query likelihood score로 word $w$의 확률을 추정하는 데 사용됨
    - 즉 document collection에 있는 모든 문서에서 단어 $w$의 확률을 추정
  - 이 relevance model은 query expansion을 위한것(relevance 모델이 유사해보이는 $P(w|R)$값이 높은 단어를 쿼리에 추가하는 방식으로 사용됨)

### Web Search
- information: 특정 주제에 대한 정보를 탐색
- navigational: 특정 웹사이트나 페이지를 찾기 위한 검색
- transactional: 특정 작업을 수행하기 위한 검색(예: 상품 구매, 예약 등)
- 상용 검색 엔진은 웹페이지의 랭킹을 위해 수백가지의 feature를 사용
  - 웹페이지의 내용, 메타데이터, 링크 구조, 사용자 행동 등 다양한 정보를 활용
    - 메타데이터: age, 갱신 빈도, URL, domain, 텍스트 길이
  - TREC 실험에서 유용했던 feature
    - title, body, heading(h1, h2, h3), anchor text, PageRank, 유입 link 수
  - 검색 유형에 따라 사용되는 feature도 달라야함
  - 쿼리를 포함한 웹페이지는 매우 많기때문에, proximity를 고려한 ranking 알고리즘이 중요

- **SEO (Search Engine Optimization)**
  - 검색시 사용되는 feature의 중요도에 따라 feature들을 조정해서 검색 결과 상위에 노출되도록 하는 작업
    - 예: title, heading, body에 키워드 포함, anchor text 최적화, 외부 링크 확보 등
    - 검색 조작에 해당할 수도 있음
    - query trap:  
      - 검색어에 해당하는 term을 아주 많이 채워 font size를 작게 하거나, 색상을 배경과 동일하게 설정하여 검색 결과에 노출되도록 하는 방법

### Machine Learning and IR
- IR 시스템에서 머신러닝을 활용하여 문서와 쿼리 간의 관련성을 학습하는 방법론이 많이 사용됨
  1. 1960년대: Rocchio 알고리즘(단순한 피드백 기반 벡터 공간 기법)
  2. 1980~90년대: 사용자 피드백 활용한 ranking 모델 개발
  3. 2000년대: text classification 기반

### Generative vs Discriminative Models
| 항목         | Generative Model (생성형 모형)                       | Discriminative Model (판별형 모형)                           |
| ------------ | ---------------------------------------------------- | ------------------------------------------------------------ |
| 기본 가정    | document와 class가 joint 확률 $P(d, c)$ 에 의해 생성 | document가 특정 class에 속할 확률 $P(c \mid d)$ 을 직접 추정 |
| 수식적 분해  | $P(d, c) = P(c) \cdot P(d \mid c)$                   | 직접 $P(c \mid d)$ 추정                                      |
| 목표         | 데이터 생성 과정을 모델링 (문서가 클래스에서 생성됨) | 클래스 구분 경계를 모델링                                    |
| 학습 방식    | 모델의 분포를 추정                                   | class 구분 문제를 직접 해결                                  |
| 데이터 요구  | 학습 데이터가 적을 때 유리                           | 데이터가 많을수록 더 유리                                    |
| IR 관점 해석 | class = query에 대한 relevant document 집합          | query-document 쌍의 feature로 relevance를 직접 학습          |
| 예시         | Naïve Bayes                                          | SVM, Logistic Regression                                     |

- Generative의 관점: 
  - 문서가 특정 class에서 생성된다고 가정
  - 문서와 class의 joint 확률을 모델링
  - 예: Naïve Bayes, Hidden Markov Model

- Discriminative의 관점:
  - 문서가 특정 class에 속할 확률을 직접 추정
  - class 구분 경계를 모델링
  - 예: SVM, Logistic Regression, Neural Networks

### Ranking Model 개발
- Ranking Model을 학습/평가하기 위해선, "정답 데이터", 즉 어느 문서가 쿼리에 대해 relevant한지에 대한 정보가 필요하다.
- 방법:  
    1. Query log에서 사용자 행동 데이터를 수집
    2. 사용자 피드백을 기반으로 relevance label 생성
    - 예: 쿼리 $Q$에 대해 문서 $D_1, D_2, \ldots, D_n$이 검색되었을 때, 사용자가 클릭한 문서 $D_i$에 대해 모든 문서 쌍에 대해 $rank(D_i) \gt rank(D_j)$로 label링
- 문서 표현 방법
    - 문서의 어떤 것을 feature로 사용할지 결정하는 것도 중요
    - 단순히 단어 빈도, proximity를 넘어 문서의 구조, 메타데이터, 링크 정보 등을 포함한 다양한 feature를 사용 가능
- feature들의 결합에 대해서도 고려해야함
    - 예: 단어 빈도와 proximity를 결합하여, 단어 빈도가 높고, 쿼리와 가까운 위치에 있는 문서가 더 높은 점수를 받도록 설정

---
해당 포스트는 서울대학교 산업공학과 박종헌 교수님의 데이터관리와 분석 25-1학기 강의를 정리한 내용입니다.  
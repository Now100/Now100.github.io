---
layout: post
title: 'Indexing and Query Processing'
date: 2025-06-06 21:16 +0900
categories: ['정보검색']
tags: ['IR', '정보검색', '강의정리']
published: true
sitemap: true
math: true
---
## Indexing
### Inverted Index
- 검색 속도를 높이기 위한 자료구조
- 각 term에 대해 해당 term이 등장하는 문서의 목록을 저장
- 문서 번호로 정렬되며, 각 entry는 posting이라고 함
- posting 안에서 특정 문서를 가리키는 pointer 사용
- 예: 
  - term: "apple"
  - posting: [(1, 1, [2]), (2, 3, [4, 5, 6]), (5, 2, [7, 8])]
    - 의미: 
        - 문서 1에서 "apple"이 1번 등장(2번 위치)
        - 문서 2에서 "apple"이 3번 등장(4, 5, 6번 위치)
        - 문서 5에서 "apple"이 2번 등장(7, 8번 위치)

- **Proximity matches**  
    - 특정 term이 서로 가까운 거리 내에 등장하는 경우를 찾는 것
    - 예: "apple"과 "banana"가 5단어 이내에 등장하는 문서 찾기
    - 이를 위해 반드시 posting list에 위치 정보가 포함되어야 함

- **Field and Extent List**
  - 문서의 구조는 검색의 정확도를 높이기 위해 중요
    - field: 문서 내의 특정 영역 (예: 제목, 본문, 저자 등)
  - extent list로 표현
    - 주어진 field type에 대해 extent(문서 내 연속된 부분)를 저장
    - inverted list는 이를 field별로 나누어 저장
    - 예: 
      - field: title
        - extent list: [(1,(0, 5)), (2,(0, 7)), (3,(0, 4))]
        - 의미: 
          - 문서 1의 title은 0부터 5까지
          - 문서 2의 title은 0부터 7까지
          - 문서 3의 title은 0부터 4까지  

- **Weight**
  - term 가중치를 문서에 대해 계산한 것을 inverted index에 저장(tf-idf 등)
    - 인덱싱 시 계산되어 저장됨
  - 검색 속도는 향상되지만, 저장 공간이 더 필요함

### Compression
- inverted index는 크기가 커질 수 있어 압축이 필요
- 압축 해제에 정보 손실이 없어야 함
- 예  
  - ambiguous encoding:  
      - 0 1 0 2 0 3이 주어짐
      - 00 01 00 10 00 11으로 표현
      - 0을 한개의 0으로 압축한다면  
        - 0 01 0 10 0 11이 되므로
        - 0/01/0/10/0/11, 0/01/01/0/0/11 두 가지로 해석될 수 있음
  - unambiguous encoding:  
      - 0: 0, 1: 101, 2: 110, 3: 111로 표현
      - 0 101 0 110 0 111로 표현
        - 이 경우 0/101/0/110/0/111으로만 해석됨

- **delta encoding**
  - 단어의 개수는 압축하기 좋음
  - 각 term의 posting list에서 문서 번호를 기준으로 차이를 저장
  - 예: 
    - 문서 번호: [1, 3, 5, 7]
    - delta encoding: [1, 2, 2, 2] (각 문서 번호에서 이전 문서 번호를 뺀 값)
    - 저장 공간을 줄일 수 있음
  - 자주 등장하는 단어는 차이값이 작아 압축 효과가 큼
  - 드물게 등장하는 단어는 차이값이 커서 압축 효과가 작음

- **byte alinged code**  
    - 컴퓨터는 바이트 단위로 데이터를 처리하므로 바이트 단위로 압축
    - v-byte: 작은 숫자에는 짧은 코드를, 큰 숫자에는 긴 코드를 사용(unicode와 유사)
      - 1바이트: 0xxxxxxx (<2^7)
      - 2바이트: 10xxxxxx 0xxxxxxx (<2^14)
      - 3바이트: 110xxxxx 10xxxxxx 0xxxxxxx (<2^21)
      - 4바이트: 1110xxxx 10xxxxxx 10xxxxxx 0xxxxxxx (<2^28)
- **Example**  
    - 다음과 같은 inverted list를 가정  
        (1, 2, [1, 7]), (2, 3, [6, 17, 197]), (3, 1, [1])
    - delta encoding:  
        - 문서 번호: [1, 2, 3] → [1, 1, 1] (각 문서 번호에서 이전 문서 번호를 뺀 값)
        - 위치: [1, 7], [6, 17, 197] [1] → [1, 6], [6, 11, 180], [1]
        - 결과: (1, 2, [1, 6]), (1, 3, [6, 11, 180]), (1, 1, [1])
    - v-byte encoding:  
        - 81 82 81 86 81 83 86 8B 01 B4 81 81 81 (16진수 인코딩)

### Skipping
- 검색은 서로 다른 inverted list의 posting을 비교하여 수행(예: "apple" AND "banana"를 검색하면, 각 term의 posting list를 비교)  
- 모든 posting을 순차적으로 비교하는 것은 비효율적
- skip pointers를 사용하여 일부 posting을 건너뛰는 방법
- 기본 알고리즘:  
  - query "apple AND banana"에 대해
    - 300개의 document가 "apple"을, 100개의 document가 "banana"를 포함한다고 가정
    - "apple"과 "banana"의 inverted list들은 document를 순차적으로 저장하고 있다고 가정
  - $d_a$: "apple"에 대한 inverted list에서 첫번째 document
  - $d_b$: "banana"에 대한 inverted list에서 첫번째 document
  - document가 존재할 때까지
    - $d_a<d_b$인 경우
      - $d_a$를 다음 document로 이동
    - $d_a>d_b$인 경우
      - $d_b$를 다음 document로 이동
    - $e_a=d_a=d_b$인 경우
      - $d_a$와 $d_b$가 같은 document를 가리키므로 결과에 추가
      - $d_a$와 $d_b$를 다음 document로 이동
- Skip pointer
  - $d_a < d_b$인 경우, "apple"의 inverted list에서 $k$만큼 건너뛰어 $s_a$로 이동
  - $s_a < d_b$인 경우, 추가로 $k$만큼 건너뛰어 $s_a'$로 이동
  - 위 과정을 반복하여 $s_a'$가 $d_b$보다 크거나 같아질 때까지 진행
  - skip pointer는 document number $d$와 byte 위치 $p$로 구성
    - 위치 $p$가 가리키는 posting은 $d$에 대한 posting
  - 예제:  
    - inverted list: [1, 4, 12, 21, 32, 33, 45, 52, 60, 120]
    - delta encoding: [1, 3, 8, 9, 11, 1, 12, 7, 8, 60]
    - skip pointer: [(12, 3), (33, 6), (60, 9)]
    - 45 찾기: (33,6)과 (60,9)를 찾을 때까지 skip pointer 검사
      - (12, 3):  
        - 12 < 45이므로 다음 posting으로 이동
      - (33, 6):  
        - 33 < 45이므로 다음 posting으로 이동
      - (60, 9):  
        - 60 > 45이므로 (33, 6)에서 다음 posting으로 이동

### 보조 도구
- Inverted file
  - inverted index를 저장하는 파일
  - term별로 posting list를 저장
  - 항상 압축 사용
  - index 업데이트가 가능해야함
- Lexicon
  - 각 term이 inverted file 내 저장된 위치를 기록
  - hash-table, B-tree 사용
- Document Stastics
  - 문서의 길이, term frequency 등 저장
  - 점수 계산에 사용(TF-IDF, BM25 등 사용)
- Distributed Indexing
  - 방대한 문서를 다루기 위해 여러 컴퓨터애 index 분산
  - 싸고 느린 여러대의 컴퓨터가 비싼 컴퓨터 한대보다 유리함
  - 병렬 indexing & query processing
  - MapReduce등

## Query Processing
- Document-at-a-time(DAAT)
  - 하나의 문서에 대해 모든 term 점수를 한번에 계산
- Term-at-a-time(TAAT)
  - 하나의 term에 대해 모든 문서 점수를 일부만 계산
  - 모든 term 처리 후 종합 점수 계산
- 두 방식 모두 최적화 기법 사용 가능

### Optimization
1. Inverted index에서 데이터를 적게 읽기
  - skip pointer 사용
  - lists를 정렬
    - PageRank 등으로 inverted list 내 document를 정렬
  - early termination
    - TAAT에서 자주 나오는 term은 무시
    - DAAT에서 상위 document만 고려
  - unsafe optimization: 최적화 전후 결과가 다를 수 있음

2. 적은 수의 document에 대해서만 scoring하기
  - conjunctive processing
    - 모든 term이 포함된 document만 scoring, 희소 term이 포함된 경우 많은 document가 제외됨
  - thresholding
    - 일정 점수 이상인 document만 scoring

- **Thresholding**(MaxScore)
  - 상위 k개 문서를 찾는 것이 목표
  - 현재 상위 k개 후보 문서 중 최소 점수를 $\tau$라고 할 때
  - 앞으로 다룰 문서중 이 $\tau$보다 높은 점수를 가질 수 없는 경우 더 계산할 필요 없음
  - safe optimization: 최적화 전후 결과가 동일함
  - 예:
    - query: "apple OR banana"
    - k = 2
    - inverted list: 
      - "apple": [1, 2, 3, 4, 5]
      - "banana": [2, 3, 5, 7]
    - indexer는 $\mu_{apple}$을 계산("apple"로만 계산 가능한 최대 점수 추정)
    - "apple"과 "banana"를 같이 포함하는 document 2개에 대한 scoring을 마쳤을 때, $\tau'$를 그 중 최소 점수라고 하면
    - $\tau' > \mu_{apple}$인 경우, "apple"만 나오는 document는 더 이상 고려할 필요 없음

### Distributed Evaluation
- 감독 컴퓨터가 모든 query 관리
- 감독 컴퓨터가 여러 index server에 query 전달
- index server들이 각각 일부분의 query를 병렬로 처리
- 각 index server는 자신의 index에서 query를 처리하고 결과를 감독 컴퓨터로 전달
- 감독 컴퓨터는 각 index server의 결과를 종합하여 최종 결과를 생성
- **Document Distribution**
  - 각 index server는 전체 코퍼스의 일정 부분에 대한 검색엔진으로 동작
  - 감독 컴퓨터는 각 index server에 query를 전달하고 각 index server는 상위 k개 문서를 반환
  - 감독 컴퓨터는 각 index server의 결과를 종합하여 최종 ranking을 생성
  - IDF와 같은 통계치는 전체 서버가 공유
- **Term Distribution**
  - 각 inverted list가 하나의 index server에 저장
  - index server 중 하나가 감독의 역할을 수행(주로 가장 긴 inverted list를 가진 서버)
  - 개별 index server의 결과들이 감독에게 전달

### Caching
- query 역시 Zipf 분포를 따름
  - 하루에 들어오는 query 중 절반만이 고유한 query
- caching은 자주 들어오는 query의 결과를 저장하여 빠르게 응답
- 많이 쓰이는 inverted list는 캐시화
  
### Information Need
- 사용자가 원하는 정보
- query는 그것이 검색엔진에 전달되는 표현
- 하지만 사용자가 원하는 정보는 query로 표현되지 못하는 경우가 많음
- 예: 
  - "apple"
    - 사용자가 원하는 정보는 "사과"일 수도, "애플 회사"일 수도 있음
    - 애플사가 아닌 "아이폰"에 대한 정보를 원할 수도 있음

#### User Interaction
- Query 입력
  - 사용자는 대부분 아주 짧은 keyword로 query를 입력
  - AND, OR, NOT 등의 연산자를 사용해 복잡한 query를 작성하기도 함
- Query 변환
  - 입력된 query를 검색엔진이 이해할 수 있는 형태로 변환
  - tokenization, stemming, stopword 제거
  - spell check, query suggestion
    - 입력 query에 대한 대안 제시
  - query expansion, relevance feedback
    - 사용자가 입력한 query에 대해 추가적인 관련 term을 찾아서 query를 확장
- 결과 출력
  - 연관된 문서들을 ranking하여 사용자에게 제공
  - snippet 생성
    - 문서의 일부를 추출하여 사용자에게 보여줌
- 시스템과의 상호작용
  - 검색 도중에도 사용자가 query를 수정하거나 추가할 수 있음
- 사용자 역할
  - 사용자는 ranking 알고리즘 자체는 못 바꾸지만, query를 조절해서 검색 결과를 바꿀 수 있음

### Stem Class
- 다양한 단어를 어근으로 묶는 방법
  - 예: "running", "ran", "runs" → "run"
- 같은 stem을 가진 단어들은 같은 의미를 가짐
- 때론 부정확한 결과를 초래할 수 있음
  - 예: "bank"와 "river bank"는 다른 의미지만 같은 stem을 가짐
- co-occurrence 정보를 사용하여 향상 가능
  - 같은 stem이어도 co-occurrence가 낮은 경우 다른 의미로 판단
- **Dice Coefficient**
  - 두 단어쌍의 동시 등장 빈도를 측정  
$$ 
\frac{2 \times |A \cap B|}{|A| + |B|}
$$
    - $|A|$와 $|B|$는 각각 단어 A와 B의 등장 횟수
    - $|A \cap B|$는 두 단어가 동시에 등장한 횟수
  - 단어쌍의 유사도를 기준으로 그래프 형성
  - 그래프에서 연결된 단어쌍을 같은 stem으로 묶음

### Spell Check
- web query의 10~15%는 오타가 있음
- 오타를 수정하여 검색 결과 향상
- "did you mean" 기능
- 기본 아이디어: 사전에 없는 단어에 대해 비슷한 단어를 찾아서 제안
- 단어 간의 유사도를 측정하여 가장 유사한 단어를 찾음
- 대표적 유사도: Edit Distance
  - 두 단어를 같게 만들기 위해 필요한 최소한의 편집 횟수
  - 편집 연산: 삽입, 삭제, 교체

- **Edit Distance**(Damerau-Levenshtein Distance)
  - 한 단어를 다른 단어로 변환하기 위한 최소 편집 횟수
  - 편집 작업
    - 삽입(Insertion): 한 글자를 추가
    - 삭제(Deletion): 한 글자를 제거
    - 교체(Substitution): 한 글자를 다른 글자로 바꿈
    - 순서 교환(Transposition): 인접한 두 글자의 순서를 바꿈
  - 예: "kitten"과 "sitting"의 edit distance는 3
    - kitten → sitten (교체)
    - sitten → sittin (교체)
    - sittin → sitting (삽입)
  - 속도 향상 기법
    - 같은 글자로 시작하는 단어만 비교
    - 같거나 유사한 길이의 단어만 비교
    - 비슷하게 들리는 단어만 비교(Soundex Code)
  - Soundex Code
    - 발음이 비슷한 단어를 같은 코드로 변환
    - 절차
      1. 단어의 첫 글자 유지
      2. 모음류(a,e,i,o,u,y,h,w)는 무시
      3. 자음은 다음과 같이 변환
         - b, f, p, v → 1
         - c, g, j, k, q, s, x, z → 2
         - d, t → 3
         - l → 4
         - m, n → 5
         - r → 6
      4. 연속된 같은 숫자는 하나로 합침
      5. 결과를 3자리수로 맞춤
    - 예: 
      - "Robert" → R163
      - "Rupert" → R163
      - "Rubin" → R150
      - "Ruben" → R150

- Spell 교정 이슈
  - 여러개의 교정 후보 중 어떤 것을 선택할지 결정해야 함
    - Context를 기반으로 선택
  - run-on 오류
    - 띄어쓰기가 잘못되어 단어가 이어지는 경우도 철자 오류로 간주

- **Noisy Channel Model**
  - $P(w|e)$: 사용자가 $e$를 입력했을 때 단어 $w$가 원래 의도한 단어일 확률
    - 이 확률을 최대화하는 단어를 찾는 것이 목표
    - $P(e|w)$: 단어 $w$를 입력했을 때 오타 $e$가 발생할 확률
    - 베이즈 정리를 사용하여  
$$
P(w|e) = \frac{P(e|w) \cdot P(w)}{P(e)}
$$
  - context를 활용한 언어 모형의 추정  
$$
P_c(w) = \lambda P(w) + (1 - \lambda) P(w|c)
$$
    - $P(w)$: 단어 $w$의 빈도
    - $P(w|c)$: context $c$에서 단어 $w$가 등장할 확률
    - $\lambda$: 두 확률의 가중치
    - 예:  
      - "apple maq"라는 query에서 "maq"는 "man"의 오타일 수도, "mac"의 오타일 수도 있지만, context로 인해 "mac"의 확률이 더 높음
  - corpus와 query로부터 언어 모델을 추정
  - error 모델
    - 단순 기법: edit distance가 같다면 같은 error의 확률을 갖는 것으로 가정
    - 복잡한 기법: 주로 발생하는 오류를 직접 모델링

- 전체 흐름
  1. 쿼리를 tokenize
  2. 각 단어마다 유사한 대안 후보 생성 (edit distance, 발음 기준 등)
  3. Noisy Channel Model로 각 후보에 대한 확률 계산
  4. 가장 가능성 높은 단어를 추천

### Query Expansion
- 검색 효율을 높이기 위해 사용자가 입력한 query에 추가적인 term을 자동/반자동으로 추가
- 자동 확장
  - 검색엔진이 자동으로 관련 term을 찾아서 query에 추가
  - 예: "apple"을 검색할 때 "fruit", "technology" 등 관련 term을 추가
- 반자동 확장
  - 사용자가 입력한 query에 대해 검색엔진이 관련 term을 제안
  - 사용자가 선택적으로 추가할 수 있음
- 단순 thesaurus 기반 확장은 효과가 낮음(context를 고려하지 않음)
- co-occurrence 기반 분석이 있어야 효과적
  - 분석 범위: corpus, 상위 문서, query log 등
- query suggetion과는 다름
  - query suggestion은 사용자가 입력한 query에 대해 대안을 제시하는 것
  - query expansion은 기존 query에 추가 term을 넣는 것

#### Term association measures
- term 간의 연관성을 측정하는 방법
- Dice Coefficent  
$$
\frac{2 \times n_{xy}}{n_x + n_y} =^{rank} \frac{n_{xy}}{n_x + n_y}
$$
- Mutual Information
  - 두 term이 얼마나 정보를 공유하는지 측정  
$$
\log \frac{P(x, y)}{P(x) P(y)} = \log N \frac{n_{xy}}{n_x n_y} =^{rank} \frac{n_{xy}}{n_x n_y}
$$
    - $P(x, y)$: x와 y가 한 슬라이딩 윈도우에서 동시에 등장할 확률
    - $P(x)$: x가 한 슬라이딩 윈도우에서 등장할 확률
    - $P(y)$: y가 한 슬라이딩 윈도우에서 등장할 확률  
  - 빈도가 낮으면 유리함
    - $n_x = n_y = 10, n_{xy} = 5$인 경우  
$$
\log \frac{5}{10 \times 10} = -2
$$
    - $n_x = n_y = 100, n_{xy} = 50$인 경우  
$$
\log \frac{50}{100 \times 100} = -4.605
$$  

- EMIM(Expected Mutual Information Measure)
  - MI를 개선하여, 확률 $P(x, y)$를 가중치로 사용해 MI의 문제점(빈도가 낮은 경우 유리함)을 해결  
$$
P(x,y)\log \frac{P(x, y)}{P(x) P(y)} = \frac{n_{xy}}{N} \log N\frac{n_{xy}}{n_x n_y}
$$
    - $N$: corpus의 크기


- Pearson Correlation Coefficient($\chi^2$)
  - 두 단어가 독립일때의 co-occurrence 기댓값과 실제 co-occurrence 기댓값의 차이를 측정  
$$
\frac{(n_{xy} - \frac{n_x n_y}{N})^2}{\frac{n_x n_y}{N}}
$$
    - $n_{xy}$: x와 y가 동시에 등장한 횟수
    - $n_x$: x가 등장한 횟수
    - $n_y$: y가 등장한 횟수
    - $N$: corpus의 크기
  - 두 term이 독립적일 때, co-occurrence는 기대값과 같음
  - 두 term이 독립적이지 않으면, co-occurrence는 기대값과 다름

- Similarity의 특성  
$$
\sigma : \mathcal{O} \times \mathcal{O} \to \mathbb{R}
$$
  - positiveness: 두 객체의 유사도는 항상 0 이상이어야 함
    - $\sigma(x, y) \geq 0,\quad \forall x, y \in \mathcal{O}$
  - maximality: 두 객체가 동일하면 유사도는 최대여야 함
    - $\sigma(x, x) \geq \sigma(x, y),\quad \forall x, y \in \mathcal{O}$
  - symmetry: 두 객체의 유사도는 순서에 상관없이 동일해야 함
    - $\sigma(x, y) = \sigma(y, x),\quad \forall x, y \in \mathcal{O}$

- Google similarity
  - document frequency를 기반으로 term 간의 유사도를 측정  
$$
sim_{Dice}(t_i, t_j) = \frac{2 \times n_{ij}}{n_i + n_j}
$$
    - $n_{ij}$: term $t_i$와 $t_j$가 동시에 등장한 횟수
    - $n_i$: term $t_i$가 등장한 횟수
    - $n_j$: term $t_j$가 등장한 횟수
  - Google 검색 결과 수를 기반으로 term 간의 유사도를 측정

- Set similarity
  - 두 term이 등장하는 document의 집합을 기반으로 유사도를 측정
  - Jaccard Similarity  
$$
sim_{Jaccard}(t_i, t_j) = \frac{|D_{t_i} \cap D_{t_j}|}{|D_{t_i} \cup D_{t_j}|}
$$
    - $D_{t_i}$: term $t_i$가 등장하는 document의 집합
    - $D_{t_j}$: term $t_j$가 등장하는 document의 집합
  - Overlap Similarity  
$$
sim_{Overlap}(t_i, t_j) = \frac{|D_{t_i} \cap D_{t_j}|}{\min(|D_{t_i}|, |D_{t_j}|)}
$$
    - 두 term이 등장하는 document의 교집합을 최소 집합 크기로 나눈 값

- Bag Similarity
  - 단어 등장 빈도로 문서를 벡터화해 유사도를 측정
  - Pearson, Cosine, Euclidean distance 등을 사용

- Phrase
  - phrase query의 경우 단일 단어 기반 확장은 비효율적
    - 예: "개그 콘서트"에 대해 "개그"와 "콘서트"를 따로 확장하면 의미가 달라짐
  - phrase 단위로 확장
  - 하지만 모든 phrase에 대해 연관 term을 찾는 것은 비현실적

- Context vector
  - 만약 한 단어가 다른 단어들과 자주 등장할 때, 해당 단어를 다른 단어들로 표현
  - 예: "apple"이 "fruit", "technology", "company"와 자주 등장하면, "apple"을 이 단어들로 표현  

- Query log
  - 사용자들이 자주 입력하는 query를 기반으로 연관 term을 찾음
  - 예: "apple"을 검색한 사용자가 "fruit", "technology" 등과 함께 검색한 경우, 이 단어들을 연관 term으로 추가
  - Query log로 유사한 query를 파악하면, query suggestion과 query expansion에 모두 사용 가능

- Personalization
  - 만약 query가 같을때 항상 같은 결과를 보여준다면
    - 사용자 개인의 선호도나 관심사를 반영하지 못함
    - 사용자가 속한 지역이나 시간대의 영향을 고려하지 못함
  - user model
    - 사용자의 검색 이력, 선호도, 관심사 등을 기반으로 개인화된 검색 결과 제공
  - query log
    - 사용자의 검색 이력을 기반으로 연관 term을 찾아서 query에 추가

- Snippets
  - 검색 결과에서 문서의 일부를 추출하여 사용자에게 보여주는 기능
  - query term은 항상 snippet에 보여줘야함
    - 하지만 이미 제목에 포함되어있다면, 본문에 다시 보여줄 필요 없음
  - URL에 query term이 포함되어 있다면, 강조표시
  - 다양한 feature들에 가중치를 부여하여 ranking 후 sippet 생성

- Sentence Selection
  - 문서에서 중요한 문장을 추출하여 snippet으로 사용
  - 문장의 중요도는 중요한 term의 빈도에 의해 계산  
$$
f_{d,w} \geq \begin{cases}
  7 - 0.1 \times (25 - s_d) & \text{if } s_d < 25 \\
  7 & \text{if } 25 \geq s_d \geq 40 \\
  7 + 0.1 \times (s_d - 40) & \text{if } s_d > 40
\end{cases}
$$
    - $s_d$: 문서 $d$의 문장 개수
    - $f_{d,w}$: 문서 $d$에서 term $w$의 빈도
      - 해당 값이 계산식보다 크거나 같으면 해당 단어는 중요 단어로 판단
  - 선택된 word sequence의 significance facotr는 중요 단어 수의 제곱을 문장 단어의 개수로 나눈 값
    - 문장이 너무 긴 경우, 중요 단어를 기준으로 인근 몇개의 단어들만 선택

- Ad
  - sponsored search
    - 광고주가 검색 결과 상단에 자신의 광고를 노출시키는 방식
  - contextual search
    - 사용자가 웹 페이지를 읽는 동안, 페이지 내용과 관련된 광고를 보여주는 방식
  - 위 두 케이스 모두 가장 연관된 광고를 보여주는 것이 중요
  - ranking시 주요 요소
    - query와 광고 text의 연관성
    - query 내 keyword의 광고 내 가격
    - 광고의 인기
  - 광고 텍스트의 특징
    - 길이가 짧음
      - 따라서 단순 단어 매칭으로는 연관성을 판단하기 어렵고, semantic matching과 query expansion이 필요
  - PRF
    - web DB가 광고 DB에 비해 매우 크므로, query expansion을 웹 검색 결과의 상위 문서들로부터 수행
  - ranking
    - 정확히 일치하는 광고를 우선
    - stem 형태로 일치하는 광고를 그 다음으로
    - query expansion을 통해 연관된 광고를 그 다음으로

- Clustering
  - 검색 결과를 클러스터링하여 유사한 문서들을 그룹화
  - 사용자는 원하는 결과를 더 쉽게 찾을 수 있음

---
해당 포스트는 서울대학교 산업공학과 박종헌 교수님의 데이터관리와 분석 25-1학기 강의를 정리한 내용입니다.  
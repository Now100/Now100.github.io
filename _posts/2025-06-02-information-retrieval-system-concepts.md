---
layout: post
title: 'Information Retrieval System Concepts'
date: 2025-06-02 20:31 +0900
categories: ['정보검색']
tags: ['IR', '정보검색', '강의정리']
published: true
sitemap: true
math: true
---
## Information Retrieval(IR)
Information Retrieval(IR, 정보 검색)은 대량의 비구조화된 데이터에서 사용자가 원하는 정보를 찾는 기술.   
정보의 구조화, 분석, 저장, 검색과 관련된 학문이다. 주로 텍스트 데이터에 적용되지만, 이미지, 오디오, 비디오 등 다양한 형태의 데이터에도 적용 가능하다.

### Document
문서(Document)는 정보 검색 시스템에서 검색의 대상이 되는 기본 단위.  

- **Database와의 차이점**
    - 구조: 
      - Database는 명확히 구조화된 데이터를 저장하고 관리하는 시스템으로, 데이터의 관계와 스키마가 명확하게 정의되어 있다.
      - Document는 비구조화된 데이터를 포함하며, 텍스트, 이미지, 오디오 등 다양한 형태로 존재할 수 있다.
    - 쿼리(Query):
      - Database는 SQL과 같은 구조화된 쿼리 언어를 사용하여 데이터를 검색한다.
      - Document는 키워드 기반의 검색이나 자연어 처리 기술을 사용하여 정보를 검색한다.
        - 자연어의 복잡성으로 인해 Document 검색은 더 복잡함

### IR의 주요 주제 및 이슈

- **주요 과제**
  - 검색(Search): 사용자가 입력한 쿼리에 대해 관련된 문서를 찾는 과정.
  - 분류(Classification): 문서를 주제나 카테고리로 분류하는 과정.
  - 필터링(Filtering): 불필요한 정보를 제거하고, 사용자가 원하는 정보만을 제공하는 과정.
  - 질의응답(Query Answering): 사용자의 질문에 대해 정확한 답변을 제공하는 과정.

- **Big issue in IR**
  - **Relevance**: 사용자가 원하는 정보와 검색 결과의 관련성을 높이는 것이 핵심 과제.
    - Topic relevance: 문서가 특정 주제와 얼마나 관련이 있는지
    - User relevance: 사용자의 정보 요구와 문서의 관련성
    - Content relevance: 문서의 내용이 사용자의 쿼리와 얼마나 일치하는지
    - Relevance model: 문서와 쿼리 간의 관련성을 수치하기 위한 모델
      - 랭킹 모델의 형태
      - 언어적 특징 외에도 통계적 특징(예: TF-IDF) 등을 활용
  - **Evaluation**: 검색 시스템의 성능을 평가하는 방법론.
    - True Positive(TP): 실제 관련된 문서이고, 검색 결과에 포함된 경우
    - False Positive(FP): 실제 관련되지 않은 문서인데, 검색 결과에 포함된 경우 **(type I error)**
    - True Negative(TN): 실제 관련되지 않은 문서이고, 검색 결과에 포함되지 않은 경우
    - False Negative(FN): 실제 관련된 문서인데, 검색 결과에 포함되지 않은 경우 **(type II error)**
    - Precision: 검색 결과 중 실제로 관련된 문서의 비율
      - Precision = TP / (TP + FP)
    - Recall: 실제 관련된 문서 중 검색 결과에 포함된 문서의 비율
      - Recall = TP / (TP + FN)
    - F1 Score: Precision과 Recall의 조화 평균
      - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    - Accuracy: 전체 검색 결과 중 실제로 관련된 문서의 비율
      - Accuracy = (TP + TN) / (TP + TN + FP + FN)

  - **Information Need**: 사용자가 정보를 검색하는 이유와 목적.
    - 쿼리만으로 사용자의 정보 요구를 완전히 이해하기 어려움(예: "apple"이 과일인지 회사인지 모호함)
    - 사용자의 정보 요구를 이해하기 위해서는 추가적인 맥락(Context)이 필요함
    - 개선 방법들:  
        - Query Expansion: 사용자의 쿼리를 확장하여 더 많은 관련 문서를 찾는 방법
        - Query Suggestion: 사용자의 쿼리를 기반으로 관련된 검색어를 제안하는 방법
        - Relevance Feedback: 사용자가 검색 결과에서 관련된 문서를 선택하도록 유도하고, 이를 기반으로 검색 결과를 개선하는 방법

### IR과 검색엔진
검색엔진은 IR의 한 형태로, 대용량 텍스트 데이터를 대상으로 정보를 검색하는 시스템.  
- **검색엔진의 초점**
    - Performance: 대량의 데이터를 빠르게 검색할 수 있는 성능이 중요함
    - Freshness: 최신 정보를 제공하기 위해 데이터의 신선도가 중요함
    - Scalability: 대량의 데이터를 처리할 수 있는 확장성이 필요함
    - Adaptability: 사용자의 정보 요구에 따라 검색 결과를 동적으로 조정할 수 있어야 함

- **검색엔진 이슈**
    - Indexing and Dynamic Data: 대량의 데이터를 효율적으로 검색하기 위해 인덱싱 기술이 필요함.  
      - 인덱스는 문서와 단어 간의 매핑을 저장하여 검색 속도를 향상시킴
      - 동적 데이터(예: 실시간 뉴스, 소셜 미디어 등)의 경우, 인덱스를 지속적으로 업데이트해야 함
      - 크롤링이 필요(웹 페이지를 자동으로 수집하는 과정)
      - 쿼리 처리와 인덱스 업데이트는 동시에 이루어져야 함
    - Spam
      - 검색 결과에 스팸(원치 않는 정보)이 포함되는 문제
      - 스팸 필터링 기술을 사용하여 관련 없는 정보를 제거함
      - 유형  
        - term spam: 특정 키워드를 과도하게 사용하여 검색 결과를 조작하는 경우
        - link spam: 다른 웹 페이지에 링크를 걸어 검색 결과를 조작하는 경우
        - SEO spam: 검색 엔진 최적화(SEO)를 악용하여 검색 결과를 조작하는 경우
      - 적대적 IR: 의도적으로 검색엔진의 알고리즘을 속이려는 시도들을 연구하는 분야

### 검색엔진 아키텍처
- 검색엔진 아키텍쳐의 요구사항
  1. Effectiveness: 사용자의 정보 요구를 충족시키는 검색 결과를 제공해야 함
  2. Efficiency: 대량의 데이터를 빠르게 검색할 수 있어야 함
- 핵심 구성요소  
    1. Indexing Process
        - 대량의 데이터를 효율적으로 검색하기 위해 인덱스를 생성하는 과정
        - 문서와 단어 간의 매핑을 저장하여 검색 속도를 향상시킴
    2. Query Process
        - 사용자의 쿼리를 처리하여 관련된 문서를 찾는 과정
        - 쿼리 확장, 쿼리 제안, 관련성 피드백 등을 포함함

#### Indexing Process
1. Text 획득
   - 웹 페이지, 문서 등 다양한 형태의 텍스트 데이터를 수집
2. Text 변환
   - 수집된 텍스트 데이터를 정제하고, 필요한 형식으로 변환
   - 예: HTML 태그 제거, 특수 문자 처리 등
3. Index 생성
   - 변환된 텍스트 데이터를 기반으로 term 기반의 인덱스를 생성
   - 인덱스는 문서와 단어 간의 매핑을 저장하여 검색 속도를 향상시킴
4. Inverted Index
   - 각 단어에 대해 해당 단어가 포함된 문서의 ID를 저장하는 구조
   - 예: "apple"이라는 단어가 포함된 문서의 ID 목록

#### Query Process
- User Interaction
   - 쿼리의 생성과 개량, 결과의 표시를 담당
- Ranking
    - query와 index에 기반하여 검색 결과를 랭킹하는 과정
- 평가: 효과와 효율성을 평가하는 과정(offline evaluation)

#### Ranking
- Scoring
  - 각 문서에 대해 점수를 계산하여 랭킹을 매김
  - 점수는 문서와 쿼리 간의 관련성을 나타냄
  - 예: $\sum(q_i \cdot d_j)$, 여기서 $q_i$는 쿼리의 i번째 단어, $d_j$는 j번째 문서의 단어(query와 document term 간의 가중치 곱)
- Ranking Features
    - Term matching: 쿼리와 문서 간의 단어 일치 여부
    - Term frequency: 문서 내에서 쿼리 단어의 빈도
    - Inverse document frequency: 쿼리 단어가 전체 문서에서 얼마나 희귀한지
    - Term proximity: 쿼리 단어 간의 거리(e.g. "white house" vs "white ... house")
    - Term position: 쿼리 단어의 위치(e.g. "apple"이 제목에 있는 경우 더 높은 점수 부여)
    - Document quality: 문서의 품질(e.g. 신뢰성, 권위성 등)
    - Web features: 웹 페이지의 링크 구조, 페이지 랭크 등

#### Evaluation
- Logging
  - 사용자 쿼리와 검색 결과를 로그로 수집해 분석
  - 클릭된 문서는 사용자가 관련성이 있다고 판단한 문서로 간주
- Ranking
  - 랭킹의 효과성 분석 및 개선
- Performance
  - 검색 속도, 인덱스 크기 등 시스템의 성능을 평가

---
해당 포스트는 서울대학교 산업공학과 박종헌 교수님의 데이터관리와 분석 25-1학기 강의를 정리한 내용입니다.  
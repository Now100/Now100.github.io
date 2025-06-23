---
layout: post
title: 'Text Processing'
date: 2025-06-02 21:56 +0900
categories: ['정보검색']
tags: ['IR', '정보검색', '강의정리', '자연어처리']
published: true
sitemap: true
math: true
---
## Text Processing
Text Processing는 정보 검색 시스템에서 텍스트 데이터를 처리하고 분석하는 기술.
### Web Crawling
웹 크롤링(Web Crawling)은 웹 페이지를 자동으로 탐색하고 데이터를 수집하는 과정.
- **WWW Process**
  - 브라우저는 웹 페이지(문서)를 요청
  - DNS 서버는 도메인 이름을 IP 주소로 변환
  - TCP 연결을 통해 웹 서버와 통신
  - HTTP 프로토콜을 사용하여 요청과 응답을 주고받음
  - 병렬 처리로 여러 페이지를 동시에 요청 가능

- **Web Crawler**
  - 웹은 매우 방대하고 지속적으로 성장
  - 웹 크롤러는 여러 웹 사이트로부터 데이터를 획득하기 위한 프로그램
    - 동적 자원들은 대개 크롤링 하지 않음
  - web site의 resource를 수집하기 위해 HTTP request를 보내고, 응답을 분석하여 필요한 정보를 추출
  - 검색엔진을 위한 크롤러는 크롤된 사이트의 변화에 대응하기위해 주기적으로 재방문을 수행해야 함
  - coverage와 freshness가 중요한 요소
    - coverage: 얼마나 많은 웹 페이지를 크롤링했는지
    - freshness: 얼마나 최근에 크롤링했는지
  - 특정 사이트 혹은 특정 주제만 크롤링 가능

- **Crawling Algorithm**
    1. request queue에 seed URL을 추가
        유명한 웹 사이트나 주제에 대한 URL을 초기 목록으로 등록
    2. request queue에 있는 URL을 하나씩 꺼내서 fetch
        - HTTP GET 요청을 통해 웹 페이지를 가져옴
    3. 응답을 파싱하여 다른 웹 페이지의 URL을 추출
    4. 추출한 URL을 request queue에 추가
    5. request queue가 비어있지 않으면 2번으로 돌아가서 반복

- **Main Issue**
  - 응답 대기시간으로 인한 비효율
    - processing 대신 thread를 사용하여 병렬 처리
  - 서버에 과부하를 주지 않도록 요청 간격 조절

#### Freshness
- 웹 페이지의 변경 여부를 확인 필요
- HTTP HEAD 요청을 통해 문서의 메타데이터를 확인할 수 있음
- Last-Modified 헤더를 통해 마지막 수정 시간을 확인
- 모든 문서에 대해 Last-Modified를 확인하는 것은 비효율적
- 자주 변경되는 페이지 위주로 확인
- Freshness & Age의 개념 도입
  - Freshness: 현 시점에서 fresh한 page의 비율
  - Age: 페이지가 마지막으로 변경된 이후 경과한 시간(내용이 변하기 전까지는 0살)

- **Age**
  - 페이지가 마지막으로 변경된 이후 경과한 시간
  - $P(x)$를 시간 $x$에서 page가 변경될 확률 밀도라고 하면
  - page가 마지막으로 crawling 된 이후 $t$일 후의 expected age는
$$
Age(\lambda, t) = \int_0^t P(x) (t - x) dx
$$
  - Poisson 분포를 가정하면  
$$
Age(\lambda, t) = \int_0^t \lambda e^{-\lambda x} (t - x) dx
$$
  - $\lambda$는 페이지가 변경될 평균 빈도(예: 하루에 한 번 변경되는 페이지라면 $\lambda = 1$)
  - 나이든 페이지일수록, crawling 안하는 비용이 커짐

### Document Processing
- Document와 Markup
  - Document: 정보 검색 시스템에서 처리되는 기본 단위, 텍스트, 이미지, 비디오 등 다양한 형태로 존재
  - Markup: Document의 구조와 의미를 정의하는 태그나 메타데이터
    - HTML, XML 등 다양한 형식으로 존재
    - Document의 구조화된 표현을 제공
    - 하나의 Document에서도 어떤 부분은 다른 부분보다 중요
    - document parser는 tag와 같은 markup을 이용해 구조 파악
      - HTML 문서에서 `<title>` 태그는 문서의 제목을 나타내고, `<h1>` 태그는 가장 중요한 제목을 나타냄
      - XML 문서에서 `<author>` 태그는 저자 정보를 나타냄
      - Bold, anchor text, header 등은 중요할 가능성이 높음
      - metadata도 중요
      - link 분석을 하기 위해 link 구조를 파악하는 것도 중요

- Document Feed
  - 대다수의 Document는 단순 발행되고 변경되지 않음(e.g. 뉴스 기사)
  - Document Feed는 하나의 Source에서 순서를 두고 발행된 Document의 모음
  - Push feed: Document가 발행될 때마다 자동으로 수신
  - Pull feed: 사용자가 주기적으로 Document를 체크하여 수신
    - RSS(Really Simple Syndication): 웹 사이트의 업데이트를 자동으로 수신하기 위한 XML 기반의 포맷, HTTP GET 요청을 통해 Document를 수신
    - XML 기반 포맷
      - Channel: Document Feed의 메타데이터(제목, 설명, 링크 등)
      - Item: 각각의 개별 컨텐츠 정보

- **Transformation Process**
  - txt, HTML, PDF 등 다양한 형식의 Document를 처리하기 위한 과정
  - 텍스트 추출, 정제, 변환 등의 작업을 포함
  - 형식 유지, 중요 포맷팅 유지, 텍스트 추출 등을 고려해야 함
  - 인코딩 변환도 중요

#### Character Encoding
- 컴퓨터가 문자를 저장/전송하는 방식
- 텍스트는 실제로는 bit의 나열
- 각 문자를 bit로 변환하는 방법이 Character Encoding
1. ASCII
   - 7비트로 128개의 문자 표현
   - 영어 알파벳, 숫자, 특수문자 등 기본적인 문자만 포함
   - 확장 ASCII는 8비트로 256개 문자 표현
2. EUC-KR
   - 한글을 포함한 문자 인코딩
   - 2바이트로 한글 문자 표현
   - 한글, 영어, 숫자, 특수문자 등 다양한 문자 지원 
3. UTF-8
    - 가변 길이 인코딩 방식
    - ASCII와 호환되며, 한글, 중국어, 일본어 등 다양한 문자 지원
    - 1~4바이트로 문자 표현
    - 국제 표준으로 널리 사용됨

#### Unicode
- 전 세계의 모든 문자를 표현하기 위한 표준
- ASCII, EUC-KR처럼 언어/국가별 인코딩 혼란 해소
- 모든 문자를 고유한 코드 포인트로 표현

| 방식   | 특징                                     |
| ------ | ---------------------------------------- |
| UTF-8  | 가변 길이, 공간 절약, 웹에 최적          |
| UTF-16 | 2 또는 4 bytes, 일부 시스템 사용         |
| UTF-32 | 고정 4 bytes, 처리 쉬움 (메모리 많이 씀) |

- **UTF-8 인코딩**
  - ASCII와 호환되며, 가변 길이 인코딩 방식
  - 1바이트부터 4바이트까지 사용하여 다양한 문자를 표현
  - 한글, 중국어, 일본어 등 다양한 문자 지원
  - 웹에서 가장 많이 사용되는 인코딩 방식
  - 예시: 
    - 그리스 문자 $\pi$s는 unicode 960번
    - 960을 2진수로 표현하면 `00000011 11000000`
    - 이 값을 UTF-8 포맷에 맞춰 변환
      - 2-byte 포맷: `110xxxxx 10xxxxxx`
      - 960 -> `00000011 11000000` -> `11000011 10111100`

| Decimal 범위  | Hex 범위         | Encoding 규칙                                   |
| ------------- | ---------------- | ----------------------------------------------- |
| 0–127         | 0x00–0x7F        | `0xxxxxxx` (1 byte) → ASCII                     |
| 128–2047      | 0x80–0x7FF       | `110xxxxx 10xxxxxx` (2 bytes)                   |
| 2048–55295    | 0x800–0xD7FF     | `1110xxxx 10xxxxxx 10xxxxxx` (3 bytes)          |
| 55296–57343   | 0xD800–0xDFFF    | Reserved (사용 안 함)                           |
| 57344–65535   | 0xE000–0xFFFF    | `1110xxxx ...`                                  |
| 65536–1114111 | 0x10000–0x10FFFF | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` (4 bytes) |

#### Document Storage
- Document를 저장하는 이유
  - 중복 크롤링 방지  
    페이지가 바뀌지 않았다면 다시 크롤링할 필요 없음
  - 정보 추출에 활용
- 저장 시스템 요구사항
  - 본문, 메타데이터, URL 등 다양한 정보 저장
  - 검색엔진을 위해 랜덤 액세스 가능(URL 기반 hash, ID 기반 접근)
  - 압축 지원
  - 버전 관리
- **Large Files**
  - document의 수가 많을떄, 하나씩 저장하면 파일, 디스크 I/O 오버헤드가 커짐
  - 여러 문서를 하나의 큰 파일로 묶어서 저장
    - 예: TREC Web 데이터는 여러 문서를 \<DOC> 태그로 묶어서 저장

#### Duplicate Detection
- 문제
  - 복제, 버전 차이, 미러링 등으로 인한 중복 문서가 많음
  - 전체 웹 페이지의 30%가 다른 페이지의 70% 이상과 동일
- Checksum
  - 문서의 내용을 해시하여 중복 여부 확인
  - 해시 값이 동일하면 내용이 동일하다고 판단
- CRC(Cyclic Redundancy Check)
  - 각 바이트의 위치 정보까지 고려하여 해시 값 생성
- near-duplicate detection
  - 문서 내용이 일부는 유사하지만 일부는 다른 경우
    - 예: 웹 페이지의 기사는 같지만 광고나 레이아웃이 다른 경우
  - document의 pair에 대해 정의된 threshold 이상으로 유사한 경우 near-duplicate로 판단
    - 예: 두 문서 D1과 D2가 포함된 단어의 90% 이상이 동일한 경우 near-duplicate로 판단
  - 검색 시나리오: document $D$의 near-duplicate를 찾는 경우
    - $O(n)$의 시간 복잡도로 모든 document와 비교
  - 발견 시나리오: 모든 document 쌍에 대해 near-duplicate를 찾는 경우
    - $O(n^2)$의 시간 복잡도로 모든 document 쌍을 비교 -> 비효율적

- **Fingerprint**
  - 문서 파싱
  - n-gram 추출
  - 일부 n-gram을 문서의 대표로 사용
  - n-gram 해시화(fingerprint)
  - fingerprint를 이용한 near-duplicate 검색
  - 예시:
    - 문서 D1: "The quick brown fox jumps over the lazy dog"
    - 2-gram 추출: "The quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog"
    - fingerprint 생성: "quick brown", "fox jumps", "lazy dog"
    - 해시화: "123", "456", "789"
    - 검색 시나리오: document $D$의 fingerprint를 이용하여 near-duplicate 검색

- **SimHash**
  - 기존의 word-based similarity 기법은 정확하지만 느림
  - SimHash는 단어 기반 유사도 + 해시 기반 fingerprint 결합 기법
  - 각 문서를 hash된 fingerprint로 표현
  - 두 fingerprint 간 bit 유사도(같은 bit의 개수)를 비교
  - 유사도는 cosine coefficient로 해석 가능
  - 알고리즘:  
    1. 문서를 단어들로 분할 후, 각 단어에 대해 등장 빈도 등을 기반으로 가중치 계산  
       - 예) "the quick brown fox jumps over the lazy dog" -> {"the": 2, "quick": 1, "brown": 1, "fox": 1, "jumps": 1, "over": 1, "lazy": 1, "dog": 1}
    2. 각 단어에 대해 b차원 hash bit 벡터 무작위 생성  
       - 예) b=8인 경우, 각 단어에 대해 8비트 벡터 생성
            - "the": [1, 0, 1, 0, 1, 0, 1, 0]
            - "quick": [0, 1, 0, 1, 0, 1, 0, 1]
            - ...
    3. 각 bit 위치에 대해 해당 단어의 i번째 bit가 1이면 해당 위치의 가중치를 더하고, 0이면 빼기. 결과는 실수값을 갖는 b차원 벡터
         - 예) 
            - "the": [1, 0, 1, 0, 1, 0, 1, 0] -> [2, -2, 2, -2, 2, -2, 2, -2]
            - "quick": [0, 1, 0, 1, 0, 1, 0, 1] -> [-1, 1, -1, 1, -1, 1, -1, 1]
            - ...
    4. $V[i] \leq 0$이면 해당 bit는 0, $V[i] > 0$이면 해당 bit는 1로 설정
       - 예) 
            - [1, -5, 3, -2, 4, -1, 2, -3] -> [1, 0, 1, 0, 1, 0, 1, 0]
    5. 최종적으로 한 문서에 대해 b비트의 SimHash 생성
  - 두 문서의 SimHash 중 같은 bit의 개수를 세어 유사도 계산

- **Noise Removal**
    - 웹 페이지에는 광고, 링크, 메뉴 등 불필요한 정보가 많음
    - 이런 노이즈를 제거하여 본문 내용만 추출하는 과정
    - Content 블록을 자동으로 추출해야함
    - 일반적으로 HTML 태그를 분석하여 본문 내용 추출
    - 대다수의 본문은 태그 수가 적고, 텍스트 밀도가 높은 곳에 위치
    - 누적 tag 수와 누적 token 수를 비교하면, text 영역이 plateau(평탄한 부분)로 나타남
    - Content 블록 찾기
        - 웹페이지를 bit의 sequence로 변환
          - $b_n = 1$ if $n$번째 token은 tag, $b_n = 0$ if $n$번째 token은 text
        - 최적화 문제로 변환
          - 구간 $[i, j]$에 대해 $b_i + b_{i+1} + ... + b_j$의 최소, $[0, i), (j, n]$에 대해 
            $b_0 + b_1 + ... + b_{i-1}$, $b_{j+1} + b_{j+2} + ... + b_n$의 최대를 구하는 문제로 변환
$$
\sum_{n=0}^{i-1} b_n + \sum_{n=i}^{j} (1-b_n) + \sum_{n=j+1}^{N-1} b_n
$$
        - 위 수식을 최대화하는 $i, j$를 찾는 문제로 변환

### Text Processing
문서를 검색어와 비교 가능한 형태로 변환하는 과정.
- **Tokenization**
  - 문서를 단어, 구, 문장 등으로 분할하는 과정
  - 언어에 따라 다르게 처리해야 함
  - 예) 영어는 공백을 기준으로 단어 분리, 한국어는 형태소 분석 필요
- **Stopping**
  - 불용어(의미 없는 단어)를 제거하는 과정
  - 예) "the", "is", "and" 등
  - 검색 효율성을 높이기 위해 불필요한 단어 제거
- **Stemming**
  - 단어의 어근을 추출하는 과정
  - 예) "running", "ran", "runs" -> "run"
  - 형태소 분석과 유사하지만, 어근 추출에 중점
- 문자열 매칭만으로는 검색이 제한적, 의미 기반 검색을 위해 처리가 필요함

#### Text Statistics
- 텍스트 데이터를 분석하여 통계 정보를 추출하는 과정.
- 단어의 등장 빈도는 예측 가능한 통계 정보
- 단어의 의미는 문서 내 등장 빈도(Occurence Frequency) 혹은 다른 단어들과 같이 등장하는 빈도(Co-occurrence Frequency)를 통해 추정 가능
- **Zipf의 법칙**
  - 단어의 출현 빈도는 심하게 편향되어 있음
  - 상위 단어 몇개가 전체 빈도의 대부분을 차지
    - 예) "the", "of", "and" 등
  - 단어들을 빈도순으로 정렬하면
$$
r \times f = k \quad \text{  or } \quad r \times P_r = c
$$
    - $r$: 단어의 순위
    - $f$: 단어의 빈도
    - $P_r$: 단어의 확률
    - $k, c$: 상수
    - 즉, 단어의 빈도와 순위는 반비례 관계
      - 순위가 2배 늘어나면 빈도는 1/2로 줄어듦
  - 응용:  
    - 특정 frequency에 해당하는 단어의 propotion은 얼마인가?
        - frequency가 $n$인 단어의 순위 $r_n = \frac{k}{n}$
        - frequency가 $n$인 단어의 수는, 빈도가 $n+1$인 단어의 순위에서 빈도가 $n$인 단어의 순위를 빼면 구할 수 있음  
$$
r_n - r_{n+1} = \frac{k}{n} - \frac{k}{n+1} = \frac{k}{n(n+1)}
$$
        - 전체 단어의 수 $\approx$ frequency가 1인 단어의 순위 = $k$
        - 따라서 frequency가 $n$인 단어의 비율은
$$
\frac{r_n - r_{n+1}}{k} = \frac{1}{n(n+1)}
$$
  
- **vocabulary의 증가**
    - Heaps Law  
      - 문서의 수가 증가함에 따라 vocabulary의 크기도 증가
      - 수식: $v = k \cdot n^\beta$
        - $v$: vocabulary의 크기
        - $n$: 문서의 수
        - $k, \beta$: 말뭉치에 따라 달라지는 상수(보통 $10\leq k \leq 100$, $\beta \approx 0.5$)
      - 큰 corpus에서는 예측이 정확하지만, 작은 corpus에서는 오차가 큼
- **검색 결과의 개수 추정**  
  - 예) query "a b c"가 모두 포함된 문서 수 예측  
$$
f_{abc} = N \cdot \frac{f_a}{N} \cdot \frac{f_b}{N} \cdot \frac{f_c}{N} = \frac{f_a \cdot f_b \cdot f_c}{N^2}
$$
  - $f_a$: 단어 a의 빈도, $N$: 전체 문서 수
  - 가정: 단어 a, b, c는 독립적으로 등장
  - 하지만 실제로는 단어 간 상관관계가 존재
  - co-occurrence를 이용하면 더 정확한 예측 가능(조건부 확률 사용)  
$$
P(a\cap b \cap c) = P(a \cap b) \cdot P(c|a\cap b)
$$
  - $P(c|a),P(c|b)$ 중 더 큰 값을 사용하여 근사
- Sampling을 이용한 추정
  - 수식: $\frac{C}{s}$
    - $C$: 샘플에서 query를 갖는 문서의 수
    - $s$: 전체 문서 중 샘플의 비율
- 전체 document corpus 크기의 추정
    - 웹 검색 엔진의 전체 document 수를 추정하는 방법
    - 수식: $N = \frac{f_a \cdot f_b}{f_{ab}}$
      - $f_a$: 단어 a의 빈도
      - $f_b$: 단어 b의 빈도
      - $f_{ab}$: 단어 a와 b가 모두 포함된 문서의 빈도

#### Tokenization
- 문서를 token(단어에 가까운 단위)으로 분할하는 과정
- 띄어쓰기, 대소문자, 구두점, 외국어 등 처리가 필요한 복잡한 작업
- 영어의 경우
  - 공백 또는 특수문자를 기준으로 단어 분리
  - 소문자화 처리 후 인덱싱
  - 구두점 제거
  - 예) "Hello, world!" -> ["hello", "world"]
- query에 적용되는 tokenization과 document에 적용되는 tokenization은 반드시 통일된 규칙을 사용해야 함

#### Stopping
- 불용어(의미 없는 단어)를 제거하는 과정
  - 불용어는 검색 결과에 큰 영향을 미치지 않지만, 검색 효율성을 높이기 위해 제거
  - 예) "the", "is", "and" 등
- "to be or not to be"와 같은 의미있는 경우는 제외
- Stopword 목록은 해당 코퍼스 내 고빈도 단어 혹은 표준 목록을 사용
- 문서 인덱싱 후 query 처리 단계에서 stopword 여부 결정

#### Stemming
- 단어의 어근을 추출하는 과정
  - 예) "running", "ran", "runs" -> "run"
- 문서와 query의 단어를 동일한 형태로 변환하여 검색 효율성 향상
- 사전 기반의 stemming과 규칙 기반의 stemming이 있음
  - 사전 기반: 단어와 어근을 매핑하는 사전을 사용
  - 규칙 기반: 접미사 제거 등 규칙을 적용하여 어근 추출
    - 예) "cats" -> "cat", "runs" -> "run"
    - false negative(의도한 질의어(supply)를 찾을 수 없음): "supplies" -> "supplie"
    - false positive(의도하지 않은 질의어(up)을 찾을 수 있음): "ups" -> "up"
    - Porter Stemmer 등 

#### Phrase
- 대부분의 query는 단어의 조합, 즉 phrase로 구성됨
- 단어 단독보다 phrase 검색이 더 정확한 결과를 제공
  - 예) "apple pie" 포함 문서 찾기 vs "apple"과 "pie" 포함 문서 찾기
- ranking 문제
  - phrase에 대해선 phrase 안의 단어들도 ranking에 포함시킬 것인지, 어떻게 우선순위를 정할 것인지 결정하기 어려운 문제가 있음
- 인식 방법
  - POS(Part of Speech) 태깅을 통해 phrase 인식
  - n-gram 모델을 사용하여 phrase 추출
  - proximity(근접성) 기반으로 phrase 인식

- **POS Tagging**
  - 문서 내 단어의 품사를 태깅하는 과정
  - 품사 태그를 기반으로 phrase 인식
  - 예) "The quick brown fox" -> [DT, JJ, JJ, NN]
  - 품사 태그를 기반으로 phrase 추출
    - 예) [DT, JJ, JJ] -> "The quick brown"
  - 태깅 후 noun phrase 추출 가능

- **n-gram 모델**
  - 문서 내 연속된 n개의 단어를 추출하여 phrase로 인식
  - 예) "The quick brown fox" -> ["The quick", "quick brown", "brown fox"]
  - POS 태깅보다 빠름
  - character n-gram 모델도 사용 가능
    - 예) "apple" -> ["ap", "pp", "pl", "le"]
  - sliding window 방식으로 n-gram 추출
  - 빈번하게 나오는 n-gram을 phrase로 인식
  - Zipf 분포를 따름
  - 특정 n값 이하의 n-gram을 모두 색인 -> 빠르지만 큰 저장공간 필요

### Link Analysis
- 웹 페이지 간의 링크 구조를 분석하여 페이지의 중요도를 평가하는 과정
- Anchor Text는 링크된 텍스트
  - 링크된 페이지의 내용을 요약하거나 설명하는 역할
  - 여러 링크들의 Anchor Text를 분석하여 페이지의 주제나 내용을 파악
  - 주로 짧고 간결한 텍스트로 구성
  - query와의 관련성 파악에 유용
- Link 분석으로 페이지 인기도, 커뮤니티 정보 추출 가능

#### PageRank
- 웹 페이지의 중요도를 평가하는 알고리즘
- link를 웹 페이지의 인기도로 간주
  - inlink(다른 페이지에서 링크된 수)가 많을수록 중요도가 높음
- link spam에 덜 민감
  - link spam: 인위적으로 페이지의 중요도를 높이기 위해 많은 링크를 생성하는 행위
- **Random Surfer Model**
  - 사용자가 웹 페이지를 무작위로 탐색하는 모델
  - 사용자는 현재 페이지에서 링크를 따라 다른 페이지로 이동하거나, 랜덤하게 다른 페이지로 이동
  - PageRank는 이 모델을 기반으로 페이지의 중요도를 계산
  - 알고리즘:  
    - 0에서 1 사이의 랜덤한 값 $r$을 생성
      - $r < \lambda$이면 랜덤하게 다른 페이지로 이동
      - $r \geq \lambda$이면 현재 페이지에서 아무 링크를 따라 이동
      - 이후 반복적으로 페이지를 탐색
  - PageRank = random surfer가 특정 페이지에 도달할 확률
    - 인기있는 페이지로부터의 link가 많을수록 PageRank가 높아짐
  - dangling link 문제
    - dangling link: 다른 페이지로 연결되지 않는 페이지, self-link만 있는 페이지, loop 형성하는 링크만 있는 페이지
    - 해결 방법: dangling link가 있는 경우 다른 페이지로 random jump를 수행

- PageRank 계산
  - 각 페이지의 PageRank는 다음과 같이 계산
$$
PR(u) = \sum_{v \in B(u)} \frac{PR(v)}{L_v}
$$
    - $PR(u)$: 페이지 $u$의 PageRank
    - $B(u)$: 페이지 $u$로 링크된 페이지들의 집합
    - $L_v$: 페이지 $v$의 outlink 수
  - 초기 PageRank 값은 모든 페이지에 대해 동일하게 설정
  - 반복적으로 PageRank 값을 업데이트하여 수렴할 때까지 계산
  - random jump를 고려: $r < \lambda$인 경우, $\frac{1}{N}$의 확률로 다른 페이지로 이동
$$
PR(u) = \frac{\lambda}{N} + (1 - \lambda) \sum_{v \in B(u)} \frac{PR(v)}{L_v}
$$

- Link quality
  - link의 품질은 spam 등의 요인으로 인해 왜곡될 수 있음
    - PageRank를 증가시키기 위한 link farm 생성
    - 유명 블로그 글에 대한 댓글에 특정 링크를 추가하여 PageRank 증가

### Information Extraction
- text에서 구조화된 정보를 추출하는 과정
  - 추출된 구조를 나타내기 위해 document에 태그 삽입
- **Named Entity Recognition (NER)**
  - 문서에서 고유명사(인물, 장소, 조직 등)를 식별하고 분류하는 작업
  - 예) "Barack Obama" -> PERSON, "New York" -> LOCATION
  - NER은 검색 엔진에서 중요한 역할을 함
    - 고유명사를 기반으로 검색 결과를 필터링하거나 정렬
    - XML-style markup을 사용하여 문서에 태그를 삽입 가능
  - 규칙 기반 기법
    - 정규 표현식, 사전 기반 매칭 등을 사용하여 고유명사 추출
    - 예) "5th Avenue" -> LOCATION, "www.example.com" -> URL
  - 통계 기반 기법
    - 단어의 빈도, co-occurrence 등을 기반으로 고유명사 추출
    - 예) "Barack Obama"가 자주 등장하는 문서에서 PERSON으로 분류

---
해당 포스트는 서울대학교 산업공학과 박종헌 교수님의 데이터관리와 분석 25-1학기 강의를 정리한 내용입니다.  
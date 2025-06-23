---
layout: post
title: 'Searching'
date: 2025-06-14 22:42 +0900
categories: ['자료구조']
tags: ['자료구조', '자료구조/알고리즘', '강의정리']
published: true
sitemap: true
math: true
---
## Searching Ordered Arrays
정렬되지 않은 배열에서는 특정 원소를 찾기 위해 배열의 모든 원소를 비교하는 선형 탐색(Linear Search)이 최선.  
하지만 정렬된 배열에서는 다양한 탐색 알고리즘을 사용할 수 있다.  

### Binary Search
Binary Search는 배열을 반으로 나누어 원하는 원소가 있는지 확인하는 방식으로 작동한다.  
1. 배열의 중간 원소를 선택한다.
2. 찾고자 하는 원소가 중간 원소보다 작으면 왼쪽 절반을, 크면 오른쪽 절반을 선택하여 재귀적으로 탐색한다.
3. 이 과정을 반복하여 원하는 원소를 찾는다.  

### Jump Search
Jump Search는 배열을 일정한 간격으로 나누어 탐색하는 방식이다.  
1. 배열 `A[0, ..., n-1]`에서 `A[k-1]`, `A[2k-1]`, ... 을 검사
2. 찾고자 하는 Key K가 `A[mk-1]`보다 작거나 같으면
   1. `A[mk-1] == K`이면 탐색 종료
   2. 그렇지 않다면 `A[(m-1)k]`부터 `A[mk-1]`까지 선형 탐색을 수행  

#### Cost
- 만약 $(m-1)k < n \leq mk$라면 최대 비교수는 
    - $m + k - 1$ (Jumps + Linear Search)
- 따라서 Jump Search의 비용  
$$
T(n, k) = \frac{n}{k} + k - 1
$$
- 비용을 최소화하는 $k$찾기  
$$
\frac{dT}{dk} = -\frac{n}{k^2} + 1 = 0 \Rightarrow k = \sqrt{n}
$$  
- 따라서 Jump Search의 최적 비용은  
$$
T(n, \sqrt{n}) = 2\sqrt{n} - 1
$$
- Jump Search는 Binary Search 보다 좋은 경우는 많지 않지만 데이터가 디스크나 느린 접근 매체에 블록 단위로 있을 때 적합.

### Interpolation Search
Interpolation Search는 Binary Search의 변형으로, 원소의 분포가 균등할 때 더 효율적이다.
- 아이디어: 찾고자 하는 값 $K$의 위치를 값을 기준으로 추정하여 탐색 범위를 좁힌다.
- 검색 위치 공식:  
$$
p = \frac{K - A[l]}{A[r] - A[l]}
$$
- 검색 위치는 $np$로 계산  

### Quadratic Binary Search(QBS)
Quadratic Binary Search는 Interpolation Search와 Jump Search를 결합한 방식이다.  
1. Interpolation Search와 같이 $p$ 계산 후 $np$ 점검
2. 만약:  
    - $A[np] < K$이면 오른쪽 절반 Jump Search
    - $A[np] > K$이면 왼쪽 절반 Jump Search
    - $A[np] = K$이면 탐색 종료
3. Jump Search:  
    - $A[np - i\sqrt{n}]$ 또는 $A[np + i\sqrt{n}]$를 검사하여 $i$ 증가
    - $K$보다 작거나 같은 최대 $i$를 찾으면 종료
4. 해당 범위 내에서 재귀적으로 QBS 수행

- 점프 후 남은 영역의 크기는 $\sqrt{n}$
- Jump Search의 probe 횟수는 상수 시간이라고 가정
- 반복적으로 동일한 QBS 수행하므로, 영역의 크기는  
$$
n \rightarrow \sqrt{n} \rightarrow \sqrt[4]{n} \rightarrow \cdots
$$
- 따라서 단계 수는 $\log \log n$개
- 최종 시간 복잡도는 $O(\log \log n)$
- 점프 당 probe 수가 상수임을 증명:  
  - $P_j$를 정확히 $j$만큼의 probe 수행이 필요한 확률로 정의
  - 최소한 2번의 probe가 필요하므로($np$ 점검 한번, Jump Search 한번) 평균 probe 수는  
$$
E[P] = 2 + \sum_{j=3}^{n} (j-2) P_j
$$
  - $P_j$의 누적 확률을 $Q_j$로 정의하면  
$$
E[P] = 2 + \sum_{j=3}^{n} Q_j
$$
  - $X$를 찾는 값 $K$보다 작은 값의 개수라고 하고, 입력 값의 분포가 균일하다고 가정하면 $X$는 binomial 분포를 따름
  - Chevychev's inequality를 사용하면  
$$
Q_j = P(X - \mu \geq (j-2) \sqrt{n}) \leq \frac{\sigma^2}{(j-2)^2n} = \frac{p(1-p)}{(j-2)^2}
$$
  - 따라서 $Q_j \leq \frac{p(1-p)}{(j-2)^2} \leq \frac{1}{4(j-2)^2}$로 상한을 잡을 수 있음  
  - 따라서
$$
E[P] \leq 2 + \sum_{j=3}^{n} \frac{1}{4(j-2)^2} = 2 + \frac{1}{4} \left( \sum_{j=1}^{n} \frac{1}{j^2} \right)
$$
  - 이떄 $\sum_{j=1}^{n} \frac{1}{j^2} = \frac{\pi^2}{6}$이므로,  
$$
E[P] \leq 2 + \frac{\pi^2}{24} \approx 2 + 0.411 \approx 2.411
$$
  - 따라서 평균 probe 수는 상수로 유지됨

## List Ordered by Frequency
각 아이템을 빈도 순으로 정렬하면 아이템의 검색 비용이 줄어듦
- 리스트를 가장 자주 접근되는 아이템부터 순서대로 배치
- 검색은 맨 앞에서부터 시작하는 선형 탐색 방식으로 진행
- 평균 탐색 시간:  
$$
\bar{C} = \sum_{i=1}^{n} \frac{f_i}{n} \cdot i
$$
  - 여기서 $f_i$는 아이템 $i$의 빈도, $n$은 아이템의 총 개수
1. 모두 동일한 빈도(uniform distribution)인 경우:  
   - $\bar{C} = \sum_{i=1}^{n} \frac{1}{n} \cdot i = \frac{1}{n} \cdot \frac{n(n+1)}{2} = \frac{n+1}{2}$
2. Geometric distribution인 경우:  
$$
\frac{f_i}{n} = p_i = \begin{cases}
    \frac{1}{2^i} \text{    if } 1 \leq i \leq n-1 \\
    \frac{1}{2^{n-1}} \text{    if } i = n
\end{cases}
$$
   - $\bar{C} \approx \sum_{i=1}^{n} \frac{i}{2^i} \approx 2$
1. Zipf distribution인 경우:  
   - Zipf distribution: $P(x) \propto x^{-1}$
   - 즉, 빈도가 높은 아이템이 빈도가 낮은 아이템보다 훨씬 더 자주 등장하는 분포
     - 상위 20%의 아이템이 전체 빈도의 80%를 차지하는 경우
       - 해당 경우의 평균 탐색 비용은 $\bar{C} \approx 0.122n$
$$
\bar{C} = \sum_{i=1}^{n} \frac{i}{i H_n} = \frac{n}{H_n} \approx \frac{n}{\log n}
$$

## Self-Organizing Lists
Self-Organizing Lists는 자주 접근되는 아이템을 리스트의 앞쪽으로 이동시켜 검색 비용을 줄이는 기법이다.
### Count Heuristic
- 각 아이템에 접근할 때마다 해당 아이템의 빈도를 증가시킴
- 빈도가 높은 아이템을 리스트의 앞쪽으로 이동시킴
- 예: 접근 패턴 `B, C, C, B, B, A`, 리스트 `A, B, C`가 있을 때,
    1. 첫번째 접근: `B`의 빈도 1, 리스트: `B, A, C`
    2. 두번째 접근: `C`의 빈도 1, 리스트: `B, C, A`
    3. 세번째 접근: `C`의 빈도 2, 리스트: `C, B, A`
    4. 네번째 접근: `B`의 빈도 2, 리스트: `C, B, A`
    5. 다섯번째 접근: `B`의 빈도 3, 리스트: `B, C, A`
    6. 여섯번째 접근: `A`의 빈도 1, 리스트: `B, C, A`
   - 최종 리스트: `B, C, A` 

### Move-To-Front Heuristic
- 아이템에 접근할 때마다 해당 아이템을 리스트의 맨 앞으로 이동시킴
- 예: 접근 패턴 `B, C, C, B, B, A`, 리스트 `A, B, C`가 있을 때,
    1. 첫번째 접근: `B`, 리스트: `B, A, C`
    2. 두번째 접근: `C`, 리스트: `C, B, A`
    3. 세번째 접근: `C`, 리스트: `C, B, A`
    4. 네번째 접근: `B`, 리스트: `B, C, A`
    5. 다섯번째 접근: `B`, 리스트: `B, C, A`
    6. 여섯번째 접근: `A`, 리스트: `A, B, C`
   - 최종 리스트: `A, B, C`

### Transpose Heuristic
- 아이템에 접근할 때마다 해당 아이템을 바로 앞의 아이템과 교환
- 예: 접근 패턴 `B, C, C, B, B, A`, 리스트 `A, B, C`가 있을 때,
    1. 첫번째 접근: `B`, 리스트: `B, A, C`
    2. 두번째 접근: `C`, 리스트: `B, C, A`
    3. 세번째 접근: `C`, 리스트: `C, B, A`
    4. 네번째 접근: `B`, 리스트: `B, C, A`
    5. 다섯번째 접근: `B`, 리스트: `B, C, A`
    6. 여섯번째 접근: `A`, 리스트: `B, A, C`

### Application - Text Compression
- Move-To-Front Heuristic을 이용한 텍스트 압축
- 단어를 리스트로 유지하고, 처음 보는 단어는 그대로 출력, 이미 등장한 단어라면 현재 위치를 출력 후 리스트의 맨 앞으로 이동
- 예시: "the car on the left hit the car I left"  

| 단어 | 리스트 상태 (처리 전)         | 출력 | 처리 후 (맨 앞으로)           |
| ---- | ----------------------------- | ---- | ----------------------------- |
| the  | \[] (처음)                    | the  | \[the]                        |
| car  | \[the]                        | car  | \[car, the]                   |
| on   | \[car, the]                   | on   | \[on, car, the]               |
| the  | \[on, car, the]               | 3    | \[the, on, car]               |
| left | \[the, on, car]               | left | \[left, the, on, car]         |
| hit  | \[left, the, on, car]         | hit  | \[hit, left, the, on, car]    |
| the  | \[hit, left, the, on, car]    | 3    | \[the, hit, left, on, car]    |
| car  | \[the, hit, left, on, car]    | 5    | \[car, the, hit, left, on]    |
| I    | \[car, the, hit, left, on]    | I    | \[I, car, the, hit, left, on] |
| left | \[I, car, the, hit, left, on] | 5    | \[left, I, car, the, hit, on] |

출력: "the car on 3 left hit 3 5 I 5"

## Searching in Sets
- Dense Set에서는(다루는 원소의 범위가 작고, 원소의 개수가 많을 때) 비트 벡터로 표현하는 것이 효율적
- 비트 벡터는 각 비트가 원소의 존재 여부를 나타내며, 검색은 비트 연산으로 수행할 수 있음
  - 예: 원소 5가 있는지 확인하려면 `A & 0b00010000`을 계산하여 0이 아닌지 확인
- 예를 들어, 1부터 15사이의 소수 집합에서 홀수를 찾는 경우, 비트 벡터를 사용하여 다음과 같이 표현할 수 있다:  
```
A = 0b0011010100010100 # 소수 집합 (인덱스 2, 3, 5, 7, 11, 13)
B = 0b0101010101010101 # 홀수 집합 (인덱스 1, 3, 5, 7, 9, 11, 13)
```
  - 이때, `A & B`를 계산하면 홀수 소수 집합을 얻을 수 있다:  
```
C = A & B = 0b0001010100010100 # 홀수 소수 집합 (인덱스 3, 5, 7, 11, 13)
```

## Hashing
Hashing은 데이터를 고정된 크기의 해시 값으로 매핑하여 검색을 빠르게 하는 기법이다.  
- 해시 함수는 입력 데이터를 고정된 크기의 해시 값으로 변환하는 함수  
$$
0 \leq h(K) \leq M-1
$$
  - 여기서 $K$는 입력 데이터, $M$은 해시 테이블의 크기
  - $h(K) = K \mod M$와 같이 간단한 해시 함수를 사용할 수 있다.
- 해시 테이블은 해시 값을 인덱스로 사용하여 데이터를 저장하는 자료구조
- 해싱 시스템의 조건은 주어진 데이터 $K$에 대해, 해시 함수의 반환값 $h(K) = i$가 해시 테이블 $HT[i]$에 저장되어야함.(즉, 상수 시간에 접근 가능)
- Collision: 해시 함수가 서로 다른 입력 데이터에 대해 동일한 해시 값을 반환하는 경우
$$ 
h(K_1) = h(K_2) \text{ for } K_1 \neq K_2
$$
- 좋은 해시 함수는 충돌을 최소화하고, 해시 테이블의 크기에 맞게 데이터를 분포시켜야 한다.  
- 예시: $h(K) = K \mod 16$   
1.  해당 해시 함수는 키가 모두 짝수인 경우, 슬롯의 절반만 사용하게 되고, 충돌이 발생할 가능성이 높아짐
2.  해당 함수는 키의 일부 비트만을 이용(마지막 4비트)하여 해시 값을 계산하기 때문에, 특정 패턴(32, 48, 64 등)으로 입력 값이 주어질 경우 충돌이 발생할 수 있음

- 소수 $M$을 사용하면 충돌을 줄일 수 있음
    - 예: $M = 17$인 경우, $h(K) = K \mod 17$는 더 균등하게 분포됨
    - 증명:  
        - 입력 값이 모두 $M$과 서로소인 어떤 상수 $c$에 대한 배수라고 가정할 때, $c \times i \equiv c \times j$라고 하면($i$와 $j$는 서로 다른 인덱스), $i \equiv j$가 되어야 함. 하지만 $0 \leq i, j < M$이므로 모순. 따라서 모든 $i$에 대해 서로 다른 해시 값을 가지게 됨.

- 다양한 비트 사용
    1. Mid-square method  
        - 입력 값의 제곱을 계산한 후, 중간 비트를 해시 값으로 사용
        - 예: $K = 1234$인 경우, $1234^2 = 1522756$이고, 중간 비트(예: 22비트)를 사용하여 해시 값을 계산
    2. Folding method  
        - 키를 바이트 단위로 나누고, 각 바이트를 더한 후 모듈러 연산을 수행해 해시 값을 계산
        - $M$이 클 경우, sum 값이 $M$을 채우지 못해 테이블 공간 낭비 가능
    3. Improved Folding method  
        - 키를 4바이트 단위로 나눠 단위값들을 더해 모듈러 연산을 수행  

### Collision Resolution
Collision Resolution은 해시 테이블에서 충돌이 발생했을 때 이를 해결하는 방법  

#### Open Hashing
- Open Hashing은 충돌된 항목을 해시 테이블 외부에 저장하는 방식이다.
- 각 해시 슬롯에 연결 리스트 등의 다른 자료구조를 사용하여 충돌된 항목을 저장한다.
- 충돌된 키들은 같은 슬롯의 리스트에 append되고, 검색 시에는 해당 리스트를 순회하여 키를 찾는다.

- **Pros**: 해시 테이블의 크기에 제한 없이 데이터를 저장할 수 있다.
- **Cons**: 해시 테이블의 크기가 커질수록 검색 시간이 증가할 수 있다. 또한, 메모리 사용량과 포인터 연산으로 인해 비용이 증가할 수 있다.

#### Closed Hashing(Open Addressing)
- Closed Hashing은 충돌된 항목을 해시 테이블 내부 다른 슬롯에 저장하는 방식이다.
- 모든 키는 고유의 home position을 가지며, 그 자리가 차있다면 다른 슬롯을 탐색(probing)하여 빈 슬롯을 찾아 저장한다.  

- **Bucket Hashing**: 해시 테이블의 각 슬롯이 고정된 크기의 버킷을 가지며, 충돌된 항목은 해당 버킷에 저장된다.
  - 해시 테이블은 여러개의 bucket으로 나뉘고, 각 bucket 내엔 여러 슬롯이 존재한다.
  - 충돌이 발생하면 해당 bucket 내의 빈 슬롯을 찾아 데이터를 저장한다.
  - bucket이 다 차면, overflow bucket을 사용한다
  - **Pros**:  
    - 테이블과 bucket만 미리 준비해두면, 충돌이 발생해도 추가적인 메모리 할당 없이 데이터를 저장할 수 있다.
  - **Cons**:  
    - bucket이 꽉 차면 overflow bucket을 사용해야 하므로, 메모리 사용량이 증가하고, 검색 시간이 증가할 수 있다. 
    - bucket 단위로 hash 위치를 정하므로, 테이블 전체에 빈 공간이 많아도 overflow bucket을 사용해야 할 수 있다.

- **Linear Probing**: 충돌이 발생하면 다음 슬롯을 순차적으로 탐색하여 빈 슬롯을 찾는다.  
$$
h(K) = (h(K) + i) \mod M
$$
  - 여기서 $i$는 충돌이 발생한 횟수로, 처음에는 $i=0$부터 시작하여 충돌이 발생할 때마다 1씩 증가한다.
  - 예: 해시 테이블의 크기가 10이고, 키 15를 저장할 때, $h(15) = 5$라면, 슬롯 5가 차있다면 6, 7, ... 순차적으로 탐색하여 빈 슬롯에 저장한다.
  - **Pros**:  
    - 구현이 간단하고, 메모리 사용량이 적다.
  - **Cons**:  
    - 클러스터링 현상으로 인해 검색 시간이 증가할 수 있다. (연속된 슬롯들이 차있으면, 다음 빈 슬롯을 찾기 위해 더 많은 슬롯을 탐색해야 함)

- **Improved Closed Hashing**: 
  - **Linear Probing with skipping**: 
    - Linear Probing에서 클러스터링 현상을 줄이기 위해, 슬롯을 일정 간격으로 건너뛰며 탐색한다.  
$$
h(K) = (h(K) + i \cdot c) \mod M
$$
    - 여기서 $c$는 건너뛰는 간격으로, 일반적으로 $M$과 서로소인 소수를 사용한다.
    - 예: $h(15) = 5$라면, 5, 7, 9, ... 순차적으로 탐색하여 빈 슬롯을 찾는다.
  - **Pseudo-Random Probing**: 
    - 해시 함수에 pseudo-random 함수를 적용하여 충돌이 발생할 때마다 다른 슬롯을 탐색한다.  
$$
h(K) = (h(K) + f(i)) \mod M
$$
    - 여기서 $f(i)$는 pseudo-random 함수로, 0부터 $M-1$ 사이의 랜덤한 순열에 대해 $i$번째 값을 반환한다.
    - 이때 랜덤한 순열은 탐색마다 다르게 생성되는 것이 아니라, 고정된 랜덤 순열을 사용한다.(해시 함수는 같은 입력에 대해 항상 같은 출력을 반환해야 하므로)
  - **Quadratic Probing**: 
    - 충돌이 발생할 때마다 탐색 간격을 제곱으로 증가시켜 탐색한다.  
$$
h(K) = (h(K) + c_1 i^2 + c_2 i + c_3) \mod M
$$
    - probe의 간격이 점점 증가하므로, 클러스터링 현상을 줄일 수 있다.

---
해당 포스트는 서울대학교 컴퓨터공학부 강유 교수님의 자료구조 25-1학기 강의를 정리한 내용입니다.
---
layout: post
title: 'Experiments, Models and Probabilities'
date: 2026-04-10 18:37 +0900
categories: ['수학', '확률론']
tags: ['확률', '모델링', '실험', 'Probability', 'Modeling', 'Experiment', 'Probability and Stochastic Processes', '내용정리']
published: true
sitemap: true
math: true
---
# 1 Experiments, Models and Probabilities
## 1.1 Set Theory
- **Set and Element**: Set(집합)은 Element(원소)들의 모임이다. 원소는 집합에 속하는 개별적인 객체를 의미한다. 예를 들어, 집합 A = {1, 2, 3}에서 1, 2, 3은 원소이다. ($1 \in A$, $2 \in A$, $3 \in A$)
- **Subset**: 집합 A의 모든 원소가 집합 B에도 포함되어 있다면, A는 B의 Subset(부분집합)이고 이를 $A \subset B$로 표현한다.
- **Set Equality**: 두 집합 A와 B가 동일한 원소를 가지고 있다면, A와 B는 같은 집합이고 이를 $A = B$로 표현한다.
- **Universal Set**: 모든 가능한 원소를 포함하는 집합을 Universal Set(보편집합)이라고 하며, 보통 $U$로 표현한다.
- **Null Set**: 원소가 하나도 없는 집합을 Null Set(공집합)이라고 하며, 보통 $\emptyset$로 표현한다.
- **Set Operations**: 집합 간의 연산은 네 가지가 있다. 
    - **Union**: 두 집합 A와 B의 Union(합집합)은 A 또는 B에 속하는 원소들의 집합으로, $A \cup B$로 표현한다.
    - **Intersection**: 두 집합 A와 B의 Intersection(교집합)은 A와 B 모두에 속하는 원소들의 집합으로, $A \cap B$로 표현한다.
    - **Difference**: 집합 A에서 B를 뺀 Difference(차집합)은 A에 속하지만 B에는 속하지 않는 원소들의 집합으로, $A - B$로 표현한다.
    - **Complement**: 집합 A의 Complement(여집합)는 보편집합 U에서 A에 속하지 않는 원소들의 집합으로, $A^c$로 표현한다.

- **Mutually Exclusive**: 여러 집합 $A_1, A_2, ..., A_n$가 있을 때, $i \neq j$인 모든 $i, j$에 대해 $A_i \cap A_j = \emptyset$이면, 이 집합들은 Mutually Exclusive(상호 배타적)이라고 한다. 즉, 서로 겹치는 원소가 없는 집합들이다. 집합이 딱 두 개일 때는 disjoint라고도 한다.
- **Collectively Exhaustive**: 여러 집합 $A_1, A_2, ..., A_n$가 있을 때, 이 집합들의 합집합이 보편집합 U와 같다면, 이 집합들은 Collectively Exhaustive(집합적으로 포괄적)이라고 한다. 

### Theorem 1.1
- De Morgan's Laws: 집합 A와 B에 대해 다음이 성립한다.
    - $(A \cup B)^c = A^c \cap B^c$
    
- Proof:
    - **$(A \cup B)^c \subset A^c \cap B^c$임을 증명**:   
        - $x \in (A \cup B)^c$라고 가정하자. 그러면 $x \notin A \cup B$이므로, $x \notin A$이고 $x \notin B$이다. 따라서 $x \in A^c$이고 $x \in B^c$이므로, $x \in A^c \cap B^c$이다.
    - **$A^c \cap B^c \subset (A \cup B)^c$임을 증명**:   
        - $x \in A^c \cap B^c$라고 가정하자. 그러면 $x \in A^c$이고 $x \in B^c$이므로, $x \notin A$이고 $x \notin B$이다. 따라서 $x \notin A \cup B$이므로, $x \in (A \cup B)^c$이다.
    - 결론적으로, $(A \cup B)^c = A^c \cap B^c$가 성립한다.

※ 두 집합이 같음을 증명하기 위해서는 두 집합이 서로의 부분집합임을 보여야 한다. 즉, $A = B$를 증명하기 위해서는 $A \subset B$와 $B \subset A$를 모두 증명해야 한다.

## 1.2 Applying Set Theory to Probability
- 확률 모델의 기본은 Procedure(절차)와 Observations로 구성된 Experiment(실험)이다. Procedure는 실험을 수행하는 방법을 정의하고, Observations는 실험에서 얻은 결과를 나타낸다.

### Definition 1.1 Outcome
> An outcome of an experiment is any possible observation of that experiment. 
- 실험의 Outcome(결과)은 그 실험에서 가능한 모든 관측값을 의미한다. 예를 들어, 동전을 던지는 실험에서 가능한 결과는 앞면이 나오는 경우와 뒷면이 나오는 경우가 있다. 따라서 이 실험의 Outcome은 {앞면, 뒷면}이다.

### Definition 1.2 Sample Space
> The sample space of an experiment is the finest-grain, mutually exclusive, collectively exhaustive set of all possible outcomes.
- 실험의 Sample Space(표본 공간)는 그 실험에서 가능한 모든 결과를 포함하는 가장 세밀한, 상호 배타적이며 집합적으로 포괄적인 집합이다. 예를 들어, 동전을 던지는 실험에서 Sample Space는 {앞면, 뒷면}이다. 주사위를 던지는 실험에서 Sample Space는 {1, 2, 3, 4, 5, 6}이다.
    - Finest-grain: 더 이상 쪼갤 수 없는 단위여야 한다. 주사위에서 "짝수/홀수"로 나누면 가장 세밀하지 않으므로 Sample Space가 될 수 없다.
    - Mutually exclusive: 두 outcome이 동시에 일어날 수 없어야 한다. 주사위에서 1이 나오면서 동시에 2가 나올 수는 없다.
    - Collectively exhaustive: 가능한 모든 경우를 빠짐없이 포함해야 한다. 주사위 Sample Space에서 6을 빼면 안 된다.

### Definition 1.3 Event
> An event is a set of outcomes of an experiment.
- 실험의 Event(사건)는 그 실험의 결과들의 집합이다. 즉, Sample Space의 부분집합이다. 예를 들어, 동전을 던지는 실험에서 "앞면이 나오는 사건"은 {앞면}이라는 집합으로 표현할 수 있다. 주사위를 던지는 실험에서 "짝수가 나오는 사건"은 {2, 4, 6}이라는 집합으로 표현할 수 있다.

### Definition 1.4 Event Space
> An event space is a collectively exhaustive, mutually exclusive set of events.
- Event Space(사건 공간)는 집합적으로 포괄적이고 상호 배타적인 사건들의 집합이다. 즉, Sample Space의 부분집합들로 이루어진 집합으로, 이 사건들이 서로 겹치지 않으면서 모든 가능한 결과를 포함해야 한다. Sample Space와 유사하지만, finest-grain 조건이 없다는 점에서 다르다. 예를 들어, 주사위를 던지는 실험에서 "짝수/홀수"로 나누면 사건 공간이 될 수 있다. 이 경우 사건 공간은 {짝수, 홀수}가 된다.

### Theorem 1.2
> For an event space $B = \{B_1, B_2, ..., B_n\}$ and any event $A$ in the same sample space, we have $A = (A \cap B_1) \cup (A \cap B_2) \cup ... \cup (A \cap B_n)$.
- Event Space B와 같은 Sample Space에 있는 어떤 사건 A에 대해서, A는 A와 B의 각 사건들의 교집합들의 합집합으로 표현할 수 있다. 즉, A는 B의 사건들과의 교집합들의 합집합으로 표현할 수 있다.
- Proof:
    - **$A \subset (A \cap B_1) \cup (A \cap B_2) \cup ... \cup (A \cap B_n)$임을 증명**:   
        - $x \in A$라고 가정하자. 그러면 $x$는 Sample Space의 원소이므로, B의 사건들 중 하나인 $B_i$에 속한다. 따라서 $x \in A$이고 $x \in B_i$이므로, $x \in A \cap B_i$이다. 따라서 $x \in (A \cap B_1) \cup (A \cap B_2) \cup ... \cup (A \cap B_n)$이다.
    - **$(A \cap B_1) \cup (A \cap B_2) \cup ... \cup (A \cap B_n) \subset A$임을 증명**:   
        - $x \in (A \cap B_1) \cup (A \cap B_2) \cup ... \cup (A \cap B_n)$라고 가정하자. 그러면 $x$는 A와 B의 어떤 사건 $B_i$의 교집합에 속한다. 즉, $x \in A$이고 $x \in B_i$이다. 따라서 $x \in A$이다.
    - 결론적으로, $A = (A \cap B_1) \cup (A \cap B_2) \cup ... \cup (A \cap B_n)$가 성립한다.

## 1.3 Probability Axioms
지금까지는 Experiments를 Procedure와 Observations로 정의하고, 이를 Sample Space, Event 등의 집합 이론적 개념으로 표현했다. 이 모델을 완성하기 위해서는 Sample Space의 모든 Event $A$에 대해 $P[A]$라는 숫자를 할당하는 것이 필요하다. 이 숫자는 Event A가 일어날 확률을 나타내며, 다음과 같은 세 가지 공리를 만족해야 한다.

### Definition 1.5 Axioms of Probability
Probability measure(확률 측도) $P[\cdot]$는 Sample Space의 모든 Event에 대해 다음 세 가지 공리를 만족하는 함수이다.
1. Axiom 1: 임의의 Event A에 대해 $P[A] \geq 0$이다. 즉, 확률은 음수가 될 수 없다.
2. Axiom 2: Sample Space $S$에 대해 $P[S] = 1$이다. 즉, Sample Space 전체의 확률은 1이다.
3. Axiom 3: Mutually Exclusive한 사건 $A_1, A_2, ...$에 대해 $P[A_1 \cup A_2 \cup ...] = P[A_1] + P[A_2] + ...$이다. 즉, 서로 겹치지 않는 사건들의 합집합의 확률은 각 사건들의 확률의 합과 같다.

확률론의 모든 이론은 오직 이 세 가지 공리에서 출발한다. 

Axiom 3를 두개의 사건으로 한정하면, 다음 정리를 얻을 수 있다.

### Theorem 1.3
> For mutually exclusive events $A$ and $B$, we have $P[A \cup B] = P[A] + P[B]$.
- 해당 정리는 Axiom 3를 보았을때 trivial해보이나, Axiom 3는 무한개의 사건에 대한 Axiom이므로, Axiom 3 그 자체로는 A와 B가 두 개의 사건인 경우에 대한 정리를 포함하지 않는다. 따라서 증명은 다음과 같이 진행된다.
- Proof:
    1. Empty set의 확률이 0임을 증명:  
        
        - $\emptyset \cap \emptyset = \emptyset$이므로, Axiom 3를 적용하여 다음과 같이 증명할 수 있다.

        $$
        P[\emptyset] = P[\sum_{i=1}^{\infty} \emptyset] = \sum_{i=1}^{\infty} P[\emptyset]
        $$
        
        - 해당 식이 성립하려면, 우변이 수렴해야하고, Axiom 1에 의해 $P[\emptyset] \geq 0$이므로, $P[\emptyset] \gt 0$이라면 우변이 발산하게 된다. 따라서 $P[\emptyset] = 0$이어야 한다.

    2. 이제 무한 수열용 규칙인 Axiom 3를 적용하기 위해, 빈자리를 채우는 사건 $C_i$를 정의하자. $C_1 = A$, $C_2 = B$, 그리고 $C_i = \emptyset$ for $i > 2$로 정의하자. 그러면 $C_i$는 Mutually Exclusive한 사건들의 집합이 된다. 이제 Axiom 2와 Axiom 3를 적용하여 다음과 같이 증명할 수 있다.

        $$
        P[A \cup B] = P[\bigcup_{i=1}^{\infty} C_i] = \sum_{i=1}^{\infty} P[C_i] = P[A] + P[B] + \sum_{i=3}^{\infty} P[\emptyset] = P[A] + P[B]
        $$

### Theorem 1.4
Theorem 1.4를 일반화하면 다음과 같이 표현할 수 있다.
> If $A = A_1 \cup A_2 \cup ... \cup A_n$ and $A_i \cap A_j = \emptyset$ for $i \neq j$, then $P[A] = P[A_1] + P[A_2] + ... + P[A_n]$.

- 즉, A가 Mutually Exclusive한 사건들의 합집합으로 표현될 수 있다면, A의 확률은 그 사건들의 확률의 합과 같다.

### Theorem 1.5
> The probability of an event $B = \{s_1, s_2, ..., s_n\}$ is the sum of the probabilities of the outcomes contained in the event:  
    $$
    P[B] = \sum_{i=1}^{n} P[s_i]
    $$

- Proof: 각 outcome $s_i$는 사건 $B$의 부분집합이고, 각 outcome은 by definition mutually exclusive하므로 다음과 같이 표현할 수 있다.
$$
B = \bigcup_{i=1}^{n} \{s_i\}
$$
Theorem 1.4를 적용하면 증명완료.

### Theorem 1.6
> For an experiment with sample space $S = \{s_1, \dots, s_n\}$ in which each outcome is equally likely,   
$$
P[s_i] = \frac{1}{n} \quad 1 \leq i \leq n
$$

- Proof: Axiom 2에 의해 $P[S] = 1$이므로, Theorem 1.5를 적용하여 다음과 같이 증명할 수 있다.
$$
1 = P[S] = P[\{s_1, s_2, ..., s_n\}] = \sum_{i=1}^{n} P[s_i] = n \cdot P[s_i]
$$
따라서 $P[s_i] = \frac{1}{n}$이 된다.

## 1.4 Some Consequences of the Axioms
확률의 Axioms에서 다양한 결과들을 도출할 수 있다. 다음은 그 중 일부이다.

### Theorem 1.7
> The probability measure $P[\cdot]$ satisfies
> (a) $P[\emptyset] = 0$
> (b) $P[A^c] = 1 - P[A]$ for any event $A$
> (c) For any events $A$ and $B$, $P[A \cup B] = P[A] + P[B] - P[A \cap B]$.
> (d) If $A \subset B$, then $P[A] \leq P[B]$.

- Proof:
    - (a) 이미 Theorem 1.3에서 증명됨.
    - (b) A와 A의 여집합 $A^c$는 Mutually Exclusive하며, $A \cup A^c = S$이므로, Axiom 2와 Theorem 1.3를 적용하여 다음과 같이 증명할 수 있다.
    $$
    1 = P[S] = P[A \cup A^c] = P[A] + P[A^c]
    $$
    따라서 $P[A^c] = 1 - P[A]$가 된다.
    - (c) A와 B의 합집합은 A와 B의 교집합을 두 번 포함하므로, 다음과 같이 표현할 수 있다.
    $$
    A \cup B = A + B - A \cap B
    $$
    Theorem 1.4를 적용하여 다음과 같이 증명할 수 있다.
    $$
    P[A \cup B] = P[A] + P[B] - P[A \cap B]
    $$
    - (d) A가 B의 부분집합이므로, 다음과 같이 표현할 수 있다.
    $$
    B = A \cup (B - A)
    $$
    Theorem 1.4를 적용하여 다음과 같이 증명할 수 있다.
    $$
    P[B] = P[A] + P[B - A]
    $$
    따라서 $P[A] \leq P[B]$가 된다.

### Theorem 1.8
> For any events $A$ and event space $B = \{B_1, B_2, ..., B_m\}$,
$$
P[A] = \sum_{i=1}^{m} P[A \cap B_i]
$$  

- Proof: Theorem 1.2에 의해 A는 B의 사건들과의 교집합들의 합집합으로 표현될 수 있다. Theorem 1.4를 적용하여 다음과 같이 증명할 수 있다.
$$
\begin{aligned}
P[A] &= P[(A \cap B_1) \cup (A \cap B_2) \cup ... \cup (A \cap B_m)] \\
&= P[A \cap B_1] + P[A \cap B_2] + ... + P[A \cap B_m]
\end{aligned}
$$

### Quiz 1.4
> 전화 통화에 대해, 누군가 말하고 있다면 음성통화($V$)이고, 모뎀이나 팩스 신호를 전달한다면 데이터통화($D$)라고 하자. 또한 통화가 3분 이상 지속된다면 긴 통화($L$), 그렇지 않다면 짧은 통화($B$)라고 하자. 수집된 데이터를 바탕으로 다음과 같은 확률이 알려져 있다고 하자.
> $P[V] = 0.7$, $P[L] = 0.6$, $P[VL] = 0.35$
> 다음 확률을 구하시오.
> (1) $P[DL]$
> (2) $P[D \cup L]$
> (3) $P[VB]$
> (4) $P[V \cup L]$
> (5) $P[V \cup D]$
> (6) $P[LB]$

- Solution:
    1. $P[DL]$을 구하기 위해, $P[L]$에서 $P[VL]$을 빼면 된다.(Theorem 1.8) 따라서 $P[DL] = P[L] - P[VL] = 0.6 - 0.35 = 0.25$가 된다.
    2. $P[D \cup L]$을 구하기 위해, Theorem 1.7 (c)를 적용하면
    $$
    P[D \cup L] = P[D] + P[L] - P[D \cap L]
    $$
    여기서 $P[D]$는 $1 - P[V] = 1 - 0.7 = 0.3$이고, $P[D \cap L]$은 $P[DL] = 0.25$이므로, 다음과 같이 계산할 수 있다.
    $$
    P[D \cup L] = 0.3 + 0.6 - 0.25 = 0.65
    $$
    3. $P[VB]$을 구하기 위해, $P[V]$에서 $P[VL]$을 빼면 된다.(Theorem 1.8) 따라서 $P[VB] = P[V] - P[VL] = 0.7 - 0.35 = 0.35$가 된다.
    4. $P[V \cup L]$을 구하기 위해, Theorem 1.7 (c)를 적용하면
    $$
    P[V \cup L] = P[V] + P[L] - P[V \cap L]
    $$
    여기서 $P[V \cap L]$은 $P[VL] = 0.35$이므로, 다음과 같이 계산할 수 있다.
    $$
    P[V \cup L] = 0.7 + 0.6 - 0.35 = 0.95
    $$
    5. $V$와 $D$는 Mutually Exclusive하므로, Theorem 1.3을 적용하면
    $$
    P[V \cup D] = P[V] + P[D] = 0.7 + 0.3 = 1.0
    $$
    6. $L$과 $B$는 Mutually Exclusive하므로, 
    $$
    P[LB] = 0 
    $$

## 1.5 Conditional Probability
- 실험을 수행하기 전, 사건 $A$의 발생 가능성에 대한 우리의 사전 지식을 나타내는 확률 $P[A]$를 Prior Probability(사전 확률)라고 한다. 실험을 수행한 후, 사건 $B$가 발생했다는 정보를 얻었다면, 사건 $A$의 발생 가능성에 대한 우리의 지식이 업데이트되어야 한다. 이때 업데이트된 확률을 Posterior Probability(사후 확률)라고 한다. 사건 $B$가 발생했다는 정보가 주어졌을 때 사건 $A$의 사후 확률을 Conditional Probability(조건부 확률)라고 한다.

### Definition 1.6 Conditional Probability
> The conditional probability of the event $A$ given the occurrence of the event $B$ is
$$
P[A|B] = \frac{P[AB]}{P[B]}
$$

- 사건 $B$가 발생했다는 정보가 주어졌을 때 사건 $A$의 확률은 사건 $A$와 $B$가 동시에 발생하는 확률을 사건 $B$의 확률로 나눈 값으로 정의된다.
- 이 정의는 $P[B] \gt 0$인 경우에만 유효하다. 
- $P[A|B]$ 역시 사건 $B$에 속하는 결과들로 이뤄진 Sample Space에서 사건 $A$가 발생할 확률로 해석할 수 있다. 즉 기존의 Axiom을 그대로 따른다.

### Theorem 1.9
> A conditional probability measure $P[A|B]$ has the following properties that correspond to the axioms of probability:
> (a) $P[A|B] \geq 0$ for any event $A$.
> (b) $P[B|B] = 1$.
> (c) If $A = A_1 \cup A_2 \cup ...$ and $A_i \cap A_j = \emptyset$ for $i \neq j$, then $P[A|B] = P[A_1|B] + P[A_2|B] + ...$.

### Theorem 1.10
> For an event space $\{B_1, B_2, ..., B_n\}$ with $P[B_i] \gt 0$ for $1 \leq i \leq n$, and any event $A$, we have
$$
P[A] = \sum_{i=1}^{n} P[A|B_i] P[B_i]
$$

- 사건 A의 확률은 사건 A가 사건 공간의 각 사건 $B_i$에서 발생할 확률과 사건 $B_i$의 확률의 곱들의 합으로 표현할 수 있다. 이를 Law of Total Probability(전체 확률의 법칙)라고 한다.
- 복잡한 사건 A의 확률을 사건 공간의 사건들을 통해 표현할 수 있기 때문에, 사건 A의 확률을 계산하는 데 도움이 된다. 

### Theorem 1.11 Bayes' Theorem
> $$ 
P[B|A] = \frac{P[A|B] P[B]}{P[A]}
$$

- 사건 A가 발생했다는 정보가 주어졌을 때 사건 B의 확률은 사건 B가 발생했을 때 사건 A의 확률과 사건 B의 확률의 곱을 사건 A의 확률로 나눈 값으로 표현할 수 있다. 이를 Bayes' Theorem(베이즈 정리)라고 한다.

- $P[A]$: Evidence
- $P[B]$: Prior Probability
- $P[B|A]$: Posterior Probability
- $P[A|B]$: Likelihood

- $B$는 주로 우리가 알고 싶은 사건이지만, $A$는 우리가 관측한 사건이 된다. 예를 들어, $B$가 "환자가 특정 질병에 걸렸는가?"라는 사건이라면, $A$는 "환자가 특정 증상을 보이는가?"라는 사건이 될 수 있다. 이 경우, $P[B|A]$는 "환자가 특정 증상을 보이는 경우에 특정 질병에 걸렸을 확률"을 나타낸다. 해당 확률을 관찰 증상의 확률(Evidence)과 질병의 확률(Prior Probability)과 질병에 걸렸을 때 그 증상이 나타날 확률(Likelihood)을 이용하여 계산할 수 있도록 하는 것이 Bayes' Theorem의 핵심이다.
- Law of Total Probability를 이용하면 다음과 같이 확장 가능  

    $$
    P[B_i|A] = \frac{P[A|B_i] P[B_i]}{\sum_{j=1}^{n} P[A|B_j] P[B_j]}
    $$

### 머신러닝 관점에서의 Bayes' Theorem
- 머신러닝은 데이터에서 패턴을 학습하여 예측 모델을 만드는 과정
- 모델의 파라미터를 $\theta$라고 하고, 관측된 데이터를 $D$라고 하자. 모델의 파라미터 $\theta$에 대한 사전 지식이 $P[\theta]$로 표현되고, 파라미터 $\theta$가 주어졌을 때 모델이 데이터를 생성할 확률이 $P[D|\theta]$로 표현된다면, Bayes' Theorem을 적용하여 모델의 파라미터 $\theta$에 대한 사후 확률을 다음과 같이 계산할 수 있다.  
$$
P[\theta|D] = \frac{P[D|\theta] P[\theta]}{P[D]}
$$
- 즉, 데이터가 주어졌을때 이 모델의 파라미터가 얼마나 가능성이 있는지를 계산할 수 있다. 이를 통해 모델의 파라미터를 업데이트하거나, 모델을 선택하는 데 활용할 수 있다.

## 1.6 Independence
- 사건 A와 B가 서로 독립적이라는 것은, 사건 A의 발생이 사건 B의 발생 확률에 영향을 미치지 않음을 의미
### Definition 1.7 Independence
> Events $A$ and $B$ are independent if and only if $P[AB] = P[A] P[B]$.
- 사건 A와 B가 0이 아닌 확률을 가질때, 위 정의는 조건부 확률 공식을 사용하면 다음 수식과 동치
$$
P[A|B] = P[A], \quad P[B|A] = P[B]
$$
- 주의할 점은 Independence(독립)은 Disjoint(서로 배타적)과 다르다는 것이다. Disjoint한 사건들은 동시에 발생할 수 없기 때문에, Disjoint한 사건 A와 B에 대해서는 $P[AB] = 0$이 된다. 따라서 Disjoint한 사건 A와 B가 모두 0이 아닌 확률을 가질 때, $P[AB] = P[A] P[B]$가 성립할 수 없다. 즉, Disjoint한 사건은 독립적일 수 없다.

### Definition 1.8 Three Independent Events
> $A_1, A_2, A_3$ are three independent events if and only if
> 1. $P[A_i A_j] = P[A_i] P[A_j]$ for $i \neq j$
> 2. $P[A_1 A_2 A_3] = P[A_1] P[A_2] P[A_3]$
- 세 사건 $A_1, A_2, A_3$가 서로 독립적이기 위해서는, 각 사건 쌍이 독립적이어야 하고, 세 사건이 동시에 발생하는 경우에도 독립적이어야 한다. 즉, 각 사건 쌍에 대해서 $P[A_i A_j] = P[A_i] P[A_j]$가 성립해야 하고, 세 사건이 동시에 발생하는 경우에도 $P[A_1 A_2 A_3] = P[A_1] P[A_2] P[A_3]$가 성립해야 한다.

### Definition 1.9 n Independent Events
> If $n \geq 3$, $A_1, A_2, ..., A_n$ are $n$ independent events if and only if
> 1. every set of $n -1$ sets taken from $A_1, A_2, ..., A_n$ is independent,
> 2. $P[A_1 A_2 ... A_n] = P[A_1] P[A_2] ... P[A_n]$.

## 1.8 Counting Methods
- 확률을 계산하기 위해서는 사건이 발생할 수 있는 경우의 수를 세는 것이 필요.

### Definition 1.10 Fundamental Principle of Counting
> If subexperiment $A$ has $n$ possivle outcomes and subexperiment $B$ has $k$ possible outcomes, then there are $n \cdot k$ possible outcomes when you perform both subexperiments.

- 동시에 두 개의 실험을 수행한다면, 가능한 결과의 수는 각 실험의 가능한 결과의 수의 곱이 된다. 예를 들어, 동전을 던지는 실험과 주사위를 던지는 실험을 동시에 수행한다면, 동전에서 가능한 결과는 2개(앞면, 뒷면)이고, 주사위에서 가능한 결과는 6개(1, 2, 3, 4, 5, 6)이므로, 두 실험을 동시에 수행할 때 가능한 결과의 수는 $2 \cdot 6 = 12$가 된다.

### Theorem 1.12
> The number of $k$-permutations of $n$ distinct objects is
$$
P(n, k) = \frac{n!}{(n-k)!}
$$

### Theorem 1.13
> The number of ways to choose $k$ objects out of $n$ distinguishable objects is
$$
\binom{n}{k} = \frac{n!}{k! (n-k)!}
$$

### Definition 1.11 n choose k
> For an integer $n \geq 0$ we define
$$
\binom{n}{k} = \begin{cases}
\frac{n!}{k! (n-k)!} & k \in \{0, 1, ..., n\} \\
0 & \text{otherwise}
\end{cases}
$$

### Theorem 1.14
> Given $m$ distinguishable objects, there are $m^n$ ways to choose with replacement an ordered sample of $n$ objects.

### Theorem 1.15
> For $n$ repetitions of a subexperiment with sample space $S = \{s_0, s_1, ..., s_{m-1}\}$, there are $m^n$ possible observation sequences.

### Theorem 1.16
> The number of observation sequences for $n$ subexperiments with sample space $S = \{0, 1\}$ with 0 appearing $n_0$ times and 1 appearing $n_1 = n - n_0$ times is $\binom{n}{n_0}$.

### Theorem 1.17
> For $n$ repetitions of a subexperiment with sample space $S = \{s_0, s_1, ..., s_{m-1}\}$, the number of length $n = n_0 + n_1 + ... + n_{m-1}$ observation sequences with $s_i$ appearing $n_i$ times for $i = 0, 1, ..., m-1$ is
$$
\binom{n}{n_0, n_1, ..., n_{m-1}} = \frac{n!}{n_0! n_1! ... n_{m-1}!}
$$

- Proof: $n$개의 위치에 $s_0$이 $n_0$개, $s_1$이 $n_1$개, ..., $s_{m-1}$이 $n_{m-1}$개 배치되어야 한다. 먼저 $n$개의 위치 중에서 $s_0$이 배치될 위치를 선택하는 방법은 $\binom{n}{n_0}$가지이다. 그 다음으로 남은 위치 중에서 $s_1$이 배치될 위치를 선택하는 방법은 $\binom{n - n_0}{n_1}$가지이다. 이 과정을 반복하여 모든 기호의 위치를 선택하면, 총 방법의 수는 다음과 같이 계산할 수 있다.  
$$
\binom{n}{n_0} \cdot \binom{n - n_0}{n_1} \cdot ... \cdot \binom{n - n_0 - n_1 - ... - n_{m-2}}{n_{m-1}} = \frac{n!}{n_0! n_1! ... n_{m-1}!}
$$

### Definition 1.12 Multinomial Coefficient
> For an integer $n \geq 0$, we define
$$
\binom{n}{n_0, n_1, ..., n_{m-1}} = \begin{cases}
\frac{n!}{n_0! n_1! ... n_{m-1}!} & n_0 + n_1 + ... + n_{m-1} = n \\
0 & \text{otherwise}
\end{cases}
$$

## 1.9 Independent Trials
- 동일한 하위 실험을 여러 번 독립적으로 반복할 때, 개별 결과가 특정 순서로 나타날 확률에 그러한 시퀀스가 나타날 수 있는 방법의 수를 곱하여 계산할 수 있다.

### Theorem 1.18
> The probability of $n_0$ failures and $n_1$ successes in $n = n_0 + n_1$ independent trials with success probability $p$ is
$$
P[S_{n_0, n_1}] = \binom{n}{n_0} p^{n_1} (1-p)^{n_0}
$$

### Theorem 1.19
> A subexperiment has sample space $S = \{s_0, s_1, ..., s_{m-1}\}$ with $P[s_i] = p_i$ for $i = 0, 1, ..., m-1$. For $n = n_0 + n_1 + ... + n_{m-1}$ trials, the probability of $n_i$ occurrences of $s_i$ for $i = 0, 1, ..., m-1$ is
$$
P[S_{n_0, n_1, ..., n_{m-1}}] = \binom{n}{n_0, n_1, ..., n_{m-1}} p_0^{n_0} p_1^{n_1} ... p_{m-1}^{n_{m-1}}
$$

## 용어정리
- Outcome: 실험에서 관찰할 수 있는 가장 개별적이고 세부적인 단일 관찰값
- Sample Space: 실험에서 가능한 모든 outcome들의 집합
- Event: Sample Space의 부분집합으로, 하나 이상의 outcome을 포함하는 집합
- Mutually Exclusive: 두 사건이 동시에 발생할 수 없는 관계(즉, 교집합이 공집합인 관계)
- Collectively Exhaustive: 여러 사건들의 합집합이 Sample Space 전체를 포함하는 관계
- Event Space: Mutually Exclusive하고 Collectively Exhaustive한 사건들의 집합, Sample Space와 유사하지만, finest-grain 조건이 없는 집합
- Probability Measure: Sample Space의 모든 사건에 대해 확률을 할당하는 함수로, 세 가지 공리를 만족해야 함
- A priori Probability: 실험을 수행하기 전에 사건의 발생 가능성에 대한 사전 지식으로 표현되는 확률
- Conditional Probability: 사건 B가 발생했다는 정보가 주어졌을 때 사건 A의 확률
- Independent Events: 사건 A와 B가 서로 독립적이라는 것은, 사건 A의 발생이 사건 B의 발생 확률에 영향을 미치지 않음을 의미
- Permutations: $n$개의 서로 다른 객체에서 $k$개를 선택하여 순서를 고려하여 배열하는 방법의 수
- Combinations: $n$개의 서로 다른 객체에서 $k$개를 선택하여 순서를 고려하지 않고 배열하는 방법의 수
- Multinomial Coefficient: 세가지 이상의 결과가 나오는 실험을 여러 번 반복할 때, 각 결과가 특정 횟수만큼 나타나는 횟수들의 경우의 수를 세는 방법

---
해당 포스트는 Yates, R. D., & Goodman, D. J. (2005). Probability and stochastic processes: A friendly introduction for electrical and computer engineers (2nd ed.). John Wiley & Sons, Inc.의 Chapter 1을 기반으로 작성되었습니다.
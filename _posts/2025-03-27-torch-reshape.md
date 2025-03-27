---
layout: post
title: 'torch.reshape()에 관하여'
date: 2025-03-27 13:32 +0900
categories: ['딥러닝', '실습']
tags: ['PyTorch', '오답노트']
published: true
sitemap: true
math: true
---
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        out, attn = self.attention(Q, K, V, mask)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.linear(out), attn
```

멀티헤드 어텐션(Multi-Head Attention)의 코드 구현을 보고 헷갈리는 부분이 있었습니다. Query, Key, Value를 구하는 부분이었는데요.

```python
Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
```
왜 `reshape(batch_size, self.num_heads, -1, self.d_k)`와 같이 `reshape()` 메서드를 활용해 텐서의 모양을 바꾸지않고 번거로이 `view()`와 `transpose()`를 사용하는지 이해가 가지 않았습니다. 그래서 직적 다음 코드를 통해 실습을 해보고 답을 얻었는데요.

```python
import torch

x = torch.tensor([
    [  # batch[0]
        [1., 2., 3., 4., 5., 6.],    # token 0
        [7., 8., 9.,10.,11.,12.],    # token 1
        [13.,14.,15.,16.,17.,18.]   # token 2
    ],
    [  # batch[1]
        [19.,20.,21.,22.,23.,24.],
        [25.,26.,27.,28.,29.,30.],
        [31.,32.,33.,34.,35.,36.]
    ]
])  # shape: (2, 3, 6)
```
위 코드는 `batch_size = 2`, `seq_len = 3`, `d_model = 6`, `num_heads = 3`인 예시입니다. 위 예제에서 우리가 바꾸고자 하는 모양은 한 시퀀스에 대한 한 헤드의 쿼리, 키, 밸류가 마지막에 오는 모양입니다. 그래야 각 헤드가 해당 시퀀스에 대한 어텐션 계산을 할 수 있을테니까요.  
그렇다면 `reshape()` 메서드를 통해 `x` 배치의 모양을 바꿔봅시다.

```python
x_reshape = x.reshape(batch_size, num_heads, seq_len, d_k)  # (2, 3, 3, 2)
print("result of reshape():")
print(x_reshaped)
```
```lua
result of reshape():
tensor([
    [  # batch 0
        [[ 1.,  2.], [ 3.,  4.], [ 5.,  6.]],
        [[ 7.,  8.], [ 9., 10.], [11., 12.]],
        [[13., 14.], [15., 16.], [17., 18.]]
    ],
    [  # batch 1
        [[19., 20.], [21., 22.], [23., 24.]],
        [[25., 26.], [27., 28.], [29., 30.]],
        [[31., 32.], [33., 34.], [35., 36.]]
    ]
])
```
우리가 원하는 것은 `[1., 2.], [ 7.,  8.], [13., 14.]`가 하나의 시퀀스로 매핑되어 첫번째 어텐션 헤드의 입력으로 들어가길 원합니다. 하지만 위와 같이 `reshape()` 메서드만으로는 모양만 맞춰줄 수 있을 뿐 원하는 결과를 만들어낼 수 없습니다.

```python
batch_size, seq_len, d_model = x.shape
num_heads = 3
d_k = d_model // num_heads

x_view = x.view(batch_size, seq_len, num_heads, d_k)  # (2, 3, 3, 2)
x_transposed = x_view.transpose(1, 2)  # (2, 3, 3, 2) → (2, 3 heads, 3 tokens, 2 d_k)
print("result of view() → transpose()")
print(x_transposed)
```
```lua
tensor([
    [  # batch 0
        [[ 1.,  2.], [ 7.,  8.], [13., 14.]],  # head 0
        [[ 3.,  4.], [ 9., 10.], [15., 16.]],  # head 1
        [[ 5.,  6.], [11., 12.], [17., 18.]]   # head 2
    ],
    [  # batch 1
        [[19., 20.], [25., 26.], [31., 32.]],
        [[21., 22.], [27., 28.], [33., 34.]],
        [[23., 24.], [29., 30.], [35., 36.]]
    ]
])
```
위와 같이 `view()`를 통해 `num_heads`만큼 `d_model`의 차원을 나눠두고, `transpose()`를 통해 우리가 원하는 방식으로 텐서를 조작할 수 있습니다.  

이런 이유를 알기위해선 `reshape()`의 동작 방식을 알아야하는데요, 우리는 고차원의 텐서를 차원대로 해석하지만, 실제 메모리에는 flat하게 저장됩니다. 
```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```
메모리: `[1, 2, 3, 4, 5, 6]`  

`reshape()` 메서드는 이렇게 flat하게 저장된 텐서를 단순히 입력받은 모양대로 끊어서 저장합니다. 따라서 `reshape()` 메서드를 복잡한 텐서에 적용하게 되면 원하는 대로 텐서를 조작하기 어려운 상황이 발생합니다. 
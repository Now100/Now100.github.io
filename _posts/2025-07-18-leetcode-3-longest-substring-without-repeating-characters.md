---
layout: post
title: 'LeetCode 3: Longest Substring Without Repeating Characters'
date: 2025-07-18 16:31 +0900
categories: ['알고리즘', '문제풀이']
tags: ['LeetCode', '슬라이딩 윈도우', '알고리즘', 'Sliding Window']
published: true
sitemap: true
math: true
---

## 문제 설명

[문제 링크](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

- 주어진 문자열에서 중복된 문자가 없는 가장 긴 부분 문자열의 길이를 구하는 문제이다.
- 문자열의 길이는 0 이상 50,000 이하이며, 각 문자는 알파벳 소문자, 대문자, 숫자, 특수 문자 등 다양한 문자가 포함될 수 있다.

## 예시

```python
Input: s = "abcabcbb"
Output: 3
Input: s = "bbbbb"
Output: 1
Input: s = "pwwkew"
Output: 3
``` 

## 풀이
문제는 슬라이딩 윈도우(Sliding Window) 기법을 사용하여 해결할 수 있다.  
슬라이딩 윈도우는 두 개의 포인터를 사용하여 현재 부분 문자열을 정의하고, 이 부분 문자열에 중복된 문자가 없는지 확인하면서 길이를 조정하는 방식이다.  
슬라이딩 윈도우를 사용하여 문자열을 순회하면서, 현재 부분 문자열에 중복된 문자가 있는지 확인하고, 중복된 문자가 발견되면 왼쪽 포인터를 이동시켜 중복을 제거한다.  
이 과정을 반복하면서 가장 긴 부분 문자열의 길이를 기록한다.  

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if s == "":
            return 0
            
        start = end = 0
        length = 1
        
        while end + 1 < len(s):
            end += 1
            while s[end] in s[start:end]:
                start += 1

            if len(s[start:end]) + 1 > length:
                length = len(s[start:end]) + 1
        
        return length
```

## 개선된 풀이
위의 풀이에서는 문자열을 순회하면서 중복된 문자를 확인하기 위해 `s[start:end]` 슬라이스를 사용했다.  
이 부분은 시간 복잡도를 $O(n^2)$로 증가시킬 수 있다.  
중복된 문자를 확인하는 데 `set`을 사용하여 시간 복잡도를 $O(1)$로 줄일 수 있다.

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = set()
        start = max_len = 0

        for end in range(len(s)):
            while s[end] in seen:
                seen.remove(s[start])
                start += 1
            seen.add(s[end])
            max_len = max(max_len, end - start + 1)

        return max_len
```
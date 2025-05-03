---
layout: post
title: 'LeetCode 33: Search in Rotated Sorted Array'
date: 2025-05-02 19:02 +0900
categories: ['알고리즘', '문제풀이']
tags: ['LeetCode', '이진탐색', '알고리즘', 'Binary Search']
published: true
sitemap: true
math: true
---


## 문제 설명

[문제 링크](https://leetcode.com/problems/search-in-rotated-sorted-array/)

- 정렬된 배열이 주어지고, 이 배열이 특정 인덱스에서 회전된 상태로 주어진다. 회전된 배열에서 특정 값을 찾는 문제이다. 배열은 중복되지 않는 정수로 이루어져 있다.  
- 배열의 길이는 1 이상 5000 이하이며, 각 원소는 $-10^4$ 이상 $10^4$ 이하이다.  
- 찾고자 하는 값은 배열에 반드시 존재한다.  
- 배열은 오름차순으로 정렬되어 있다.  
- 회전된 배열은 원래 배열의 일부를 앞쪽으로 이동시킨 형태이다. 예를 들어, [0,1,2,4,5,6,7] 배열이 [4,5,6,7,0,1,2]로 회전된 경우이다.
- $O(\log n)$의 시간 복잡도로 해결해야 한다.

## 예시

```python
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Input: nums = [1], target = 0
Output: -1
```

## 풀이

$O(\log n)$의 시간 복잡도로 해결하기 위해 이진 탐색을 사용하였다.  
문제의 조건상 왼쪽 배열에 target이 존재하는 경우를 정리해보았다.  

- `nums[left] <= target < nums[mid]`: 왼쪽 배열이 순서대로 정렬되어 있고, target이 왼쪽 배열에 존재하는 경우이다.   

- `nums[mid] < nums[left]`: 회전축(pivot)이 왼쪽 배열에 존재하는 경우이다.
  - `nums[left] <= target`: target이 반드시 회전축 이전에 존재하므로, 왼쪽 배열에 target이 존재한다.
    - 예시)
        - `nums = [5,6,0,1,2,3,4]`
        - `target = 6`
        - `left = 0`, `right = 6`, `mid = 3`
        - `nums[left] = 5`, `nums[mid] = 1`, `target = 6`
  - `target < nums[mid]`: target이 회전축과 mid 사이에 존재하므로, 왼쪽 배열에 target이 존재한다.
    - 예시)
        - `nums = [5,6,0,1,2,3,4]`
        - `target = 0`
        - `left = 0`, `right = 6`, `mid = 3`
        - `nums[left] = 5`, `nums[mid] = 1`, `target = 0`

이 경우에는 왼쪽 배열에 target이 존재하므로, `right = mid - 1`로 설정한다.  

이외의 경우에는 오른쪽 배열에 target이 존재하므로, `left = mid + 1`로 설정한다.  

## 코드

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        n = len(nums)
        l = 0
        r = n-1

        while(l<=r):
            mid = (l+r) // 2

            if target == nums[mid]:
                return mid
            elif (nums[l] <= target and target < nums[mid]) or \
            (nums[mid] < nums[l] and (nums[l] <= target or target < nums[mid])):
                r = mid - 1
            else:
                l = mid + 1

        return -1
```

## 다른 풀이
- 이진 탐색을 두번 수행하여, 첫번째 이진 탐색으로 회전축(pivot)을 찾고, 두번째 이진 탐색으로 target을 찾는 방법이다.  

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        n = len(nums)
        l = 0
        r = n-1

        # Find pivot
        while(l<r):
            mid = (l+r) // 2

            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid

        pivot = l
        l = 0
        r = n-1

        # Find target
        while(l<=r):
            mid = (l+r) // 2

            if target == nums[(mid + pivot) % n]:
                return (mid + pivot) % n
            elif nums[(mid + pivot) % n] < target:
                l = mid + 1
            else:
                r = mid - 1

        return -1
```  
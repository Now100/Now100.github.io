---
layout: post
title: 'MaxRetryError 해결'
date: 2025-03-17 11:22 +0900
categories: ['크롤링']
tags: ['오답노트', '크롤링']
published: true
sitemap: true
math: true
---
```python
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

driver = webdriver.Chrome()

# 정상 동작
driver.get("https://www.google.com")

# 드라이버 종료
driver.quit()

# 종료된 드라이버에서 작업 수행 (예상 오류 발생)
try:
    driver.get("https://www.google.com")
except WebDriverException as e:
    print(f"Error: {e}")
```
`MaxRetryError: HTTPConnectionPool(host='localhost', port=1234):Max retries exceeded with url: /session (Caused by NewConnectionError)`

위와 같이 종료된 드라이버를 사용해 다시 크롤링을 하려 하면 이와 같은 오류를 얻을 수 있다.  
처음엔 너무 많은 리퀘스트를 보내서 해당 오류가 뜬 줄 알았는데 알고보니 `quit()`으로 종료한 드라이버를 사용해 그런 것이었다.  
다시 드라이버 인스턴스를 할당하고 크롤링하면 정상작동한다. 
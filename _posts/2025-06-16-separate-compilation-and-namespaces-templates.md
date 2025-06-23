---
layout: post
title: 'Separate Compilation and Namespaces, Templates'
date: 2025-06-16 22:09 +0900
categories: ['C++']
tags: ['C++', '강의정리']
published: true
sitemap: true
math: true
---
## Separate Compilation 
C++에서 **Separate Compilation**은 프로그램을 여러 개의 소스 파일로 나누어 컴파일하는 방법. 컴파일된 파일들은 링커를 통해 하나의 실행 파일로 결합됨. 이 방법은 코드의 모듈화와 재사용성을 높이고, 대규모 프로젝트에서 컴파일 시간을 단축시킴.  
클래스를 정의할때는 정의 부분과 구현 부분을 분리하여 작성할 수 있음. 이렇게 하면 라이브러리화 하여 다른 프로그램에서도 재사용할 수 있음.
- **Encapsulation**: OOP의 핵심 원칙 중 하나로, 클래스의 내부 구현이 변경되더라도 클래스 사용자 코드는 영향을 받지 않도록 함.
- 파일 구성
  - **헤더 파일**(`.h`): 클래스의 선언부를 포함하며, 클래스의 인터페이스를 정의함.
  - **소스 파일**(`.cpp`): 클래스의 구현부를 포함하며, 클래스의 메서드 정의를 포함함. 반드시 해당하는 헤더 파일을 포함해야 함.(`#include "ClassName.h"`)

- **예시**: `Point` 클래스의 헤더 파일과 소스 파일  
```cpp
// Point.h
#ifndef POINT_H
#define POINT_H
class Point {
    public:
        Point(int xVal, int yVal);
        int getX() const;
        int getY() const;
        void setX(int xVal);
        void setY(int yVal);
    private:
        int x; // x 좌표
        int y; // y 좌표
};
#endif // POINT_H
```
```cpp
// Point.cpp
#include "Point.h"
Point::Point(int xVal, int yVal) : x(xVal), y(yVal) {}
int Point::getX() const { return x; }
int Point::getY() const { return y; }
void Point::setX(int xVal) { x = xVal; }
void Point::setY(int yVal) { y = yVal; }
```
- 헤더파일 다중 포함 방지
  - 헤더 파일이 여러 번 포함되는 것을 방지하기 위해 `#ifndef`, `#define`, `#endif`를 사용하여 헤더 파일의 중복 포함을 방지함.  
  - `#pragma once`를 사용하여 헤더 파일이 한 번만 포함되도록 할 수도 있음. 이 방법은 컴파일러에 따라 다르게 동작할 수 있지만, 대부분의 현대 컴파일러에서 지원됨.

```cpp
// A.h
class A{};

// B.h
#include "A.h"
class B{};

// main.cpp
#include "A.h"
#include "B.h" // 중복 포함 발생 -> 중복 정의 오류
```

## Namespaces
C++에서 **네임스페이스(Namespace)**는 이름 충돌을 방지하기 위해 사용되는 기능. 서로 다른 라이브러리나 모듈에서 동일한 이름의 함수나 변수가 정의될 때, 네임스페이스를 사용하여 이름 충돌을 방지할 수 있음.
- 네임스페이스는 `namespace` 키워드를 사용하여 정의됨
- **예시**:  

```cpp
namespace MyNamespace {
    int x;
    void myFunction() {
        // 함수 구현
    }
}
```
- Scope
  - global namespace: 코드는 기본적으로 global namespace에 속함. 즉, 네임스페이스를 명시하지 않으면 global namespace에 정의된 것으로 간주됨.
  - local namespace: 함수나 클래스 내부에서 정의된 네임스페이스는 해당 함수나 클래스의 로컬 네임스페이스로 간주됨. 이 경우, 해당 네임스페이스는 해당 함수나 클래스 내부에서만 유효함.
  - 만약 같은 이름을 쓰는 namespace가 여러개 있다면, 같은 scope에 있다면 충돌이 발생함.
  - scope는 {} 단위  

```cpp
namespace NM1{ int func() { return 1; } }
namespace NM2{ int func() { return 2; } }

int main() {
    {
        using namespace NM1; // NM1 네임스페이스 사용
        int result = func(); // NM1::func() 호출
    }
    {
        using namespace NM2; // NM2 네임스페이스 사용
        int result = func(); // NM2::func() 호출
    }
    return 0;
}
```

- nested namespace: 네임스페이스 안에 또 다른 네임스페이스를 정의할 수 있음.    

```cpp
namespace OuterNamespace {
    namespace InnerNamespace {
        int value = 42;
        void innerFunction() {
            // 함수 구현
        }
    }
}
// OuterNamespace::InnerNamespace::value로 접근 가능
```

### 사용법
1. using directive    

```cpp
using namespace MyNamespace; // MyNamespace의 모든 이름을 현재 스코프로 가져옴
int main() {
    x = 10; // MyNamespace의 x에 접근
    myFunction(); // MyNamespace의 myFunction 호출
    return 0;
}
```
2. using declaration    

```cpp
using MyNamespace::x; // MyNamespace의 x만 현재 스코프로 가져옴
int main() {
    x = 10; // MyNamespace의 x에 접근
    // myFunction(); // 오류 발생: myFunction은 현재 스코프에 없음
    return 0;
}
```
3. Fully qualified name  

```cpp
int main() {
    MyNamespace::x = 10; // MyNamespace의 x에 접근
    MyNamespace::myFunction(); // MyNamespace의 myFunction 호출
    return 0;
}
```

## Templates
C++에서 **템플릿(Template)**은 함수나 클래스를 일반화하여 다양한 데이터 타입에 대해 재사용할 수 있도록 하는 기능. 템플릿을 사용하면 코드의 중복을 줄이고, 타입에 구애받지 않는 유연한 코드를 작성할 수 있음.  
타입은 컴파일 타임에 결정되며, 템플릿을 사용하여 함수나 클래스를 정의할 때는 `template` 키워드를 사용함.  
컴파일러는 필요할 때마다 템플릿을 해당 타입에 맞는 코드에 맞춰 인스턴스화함.

- **함수 템플릿**: 함수의 매개변수 타입을 일반화하여 다양한 타입의 인자를 받을 수 있도록 함.  

```cpp
void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}
```
- 위 함수는 `int` 타입에 대해서만 동작함. 이를 다양한 자료형에 대해 일반화하려면 템플릿을 사용함.  

```cpp
template <typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}
int a = 10, b = 20;
swap(a, b); // int 타입에 대해 swap 호출
double x = 1.5, y = 2.5;
swap(x, y); // double 타입에 대해 swap 호출
```
- `template <typename T>`은 `template <class T>`로도 표현할 수 있음. 여기서 `class`는 어떤 타입을 나타내는지에 대한 약속일 뿐, 실제로 클래스 타입만을 의미하지 않음.  
- 오버로딩은 데이터 타입에 따라 함수 이름을 다르게 정의하는 것이지만, 템플릿은 하나의 함수 이름으로 다양한 데이터 타입에 대해 동작할 수 있도록 함.
- 템플릿은 항상 모든 인자 타입에 적용되는 것은 아님  

```cpp
template <typename T>
void func(int x, T y, T z) {
    // ...
}
```
- 함수의 선언과 정의를 다른 파일에 분리할 때가 있지만, 템플릿 함수는 컴파일 타임에 인스턴스화되기 때문에, 템플릿 함수는 선언부와 정의부를 같은 파일에 작성해야 함.   

```cpp
// MyTemplate.h
#ifndef MYTEMPLATE_H
#define MYTEMPLATE_H
template <typename T>
void print(const T& value) {
    std::cout << value << std::endl;
}
#endif // MYTEMPLATE_H
```  
- 다중 타입에 대해 동작하는 함수 템플릿을 정의할 때는 `template <typename T1, typename T2>`와 같이 여러 타입 매개변수를 사용할 수 있음.
  - 사용되지 않은 타입 매개변수는 컴파일러가 경고를 발생시킬 수 있음.  

```cpp
template <typename T1, typename T2>
void printPair(const T1& first, const T2& second) {
    std::cout << "First: " << first << ", Second: " << second << std::endl;
}
int main() {
    printPair(10, 20.5); // int와 double 타입에 대해 호출
    printPair("Hello", 42); // string과 int 타입에 대해 호출
    return 0;
}
```
- **클래스 템플릿**: 클래스의 멤버 변수나 메서드의 타입을 일반화하여 다양한 타입의 객체를 생성할 수 있도록 함.  

```cpp
template <typename T>
class Pair{
    public:
        Pair(T first, T second) : first(first), second(second) {}
        
        T getFirst() const { return first; }
        T getSecond() const { return second; }
        
        void setFirst(T value) { first = value; }
        void setSecond(T value) { second = value; }
        
    private:
        T first;  // 첫 번째 값
        T second; // 두 번째 값
};
```
- 위 클래스 템플릿은 `T` 타입의 값을 저장하고, 해당 값을 반환하거나 설정하는 메서드를 제공함.
- 주의점: 클래스 템플릿의 모든 멤버함수 정의에도 `template <typename T>`를 명시해야 함.    

```cpp
template <typename T>
T Pair<T>::getFirst() const {
    return first;
}
template <typename T>
T Pair<T>::getSecond() const {
    return second;
}
template <typename T>
Pair<T>::Pair() {...} // Constructor 정의

template <typename T>
Pair<T>::~Pair() {...} // Destructor 정의
```
- `Pair<T>`와 같이 템플릿 클래스임을 명시해야 컴파일러가 올바르게 인식함.
- 클래스 템플릿은 타입을 명시하면 함수의 파라미터로 사용할 수 있음.  

```cpp
int sumPair(const Pair<int>& p) {
    return p.getFirst() + p.getSecond();
}
```
- **함수 템플릿 & 클래스 템플릿**: 함수 템플릿과 클래스 템플릿을 함께 사용할 수 있음.  

```cpp
template <typename T>
T sumPair(const Pair<T>& p) {
    return p.getFirst() + p.getSecond();
} // + 연산이 가능한 T 타입에 대해서만 동작함.
```
- Predefined Template Types: C++ STL에서 제공하는 템플릿 타입을 사용할 수 있음. 예를 들어, `std::vector`, `std::list`, `std::map` 등은 모두 템플릿 클래스로 구현되어 있음.  

```cpp
#include <vector>
#include <iostream>
int main() {
    std::vector<int> vec; // int 타입의 벡터
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    
    for (const auto& val : vec) {
        std::cout << val << " "; // 1 2 3 출력
    }
    return 0;
}
```
- `std::string`은 사실 `std::basic_string<T>`의 특수화로, `T`가 `char`인 경우에 해당함.  

---
해당 포스트는 서울대학교 전기정보공학부 정교민 교수님의 프로그래밍방법론 25-1학기 강의를 정리한 내용입니다.
---
layout: post
title: 'Constructors & Other Tools'
date: 2025-06-16 16:42 +0900
categories: ['C++']
tags: ['C++', '강의정리']
published: true
sitemap: true
math: true
---
## Constructors
C++에서 **생성자(Constructor)**는 클래스 객체 초기화를 위해 사용되는 특별한 멤버함수. 멤버 변수 일부 또는 전체를 초기화하는 데 사용되며, 객체가 생성될 때 자동으로 호출됨. 생성자는 클래스 이름과 동일한 이름을 가지며, 반환 타입이 없음.  
클래스 이름과 동일한 이름을 가져아햐고, 반드시 public 영역에 선언되어야 함. 생성자는 매개변수를 가질 수 있으며, 기본 생성자와 매개변수가 있는 생성자를 모두 정의할 수 있음.  

- **기본 생성자**: 매개변수가 없는 생성자로, 객체가 생성될 때 자동으로 호출되어 멤버 변수를 기본값으로 초기화함.   

```cpp
class Point {
  public:
      Point(){
          x = 0; // 기본값으로 x 좌표를 0으로 초기화
          y = 0; // 기본값으로 y 좌표를 0으로 초기화
      }
      print() {
          cout << "Point(" << x << ", " << y << ")" << endl; // 현재 좌표 출력
      }
  private:
      int x; // x 좌표
      int y; // y 좌표
};

int main() {
    Point p; // Point 객체 생성 시 기본 생성자가 호출되어 x, y가 0으로 초기화됨
    p.print(); // Point(0, 0) 출력
    return 0;
}
```

- **매개변수가 있는 생성자**: 매개변수를 받아 멤버 변수를 초기화하는 생성자. 객체를 생성할 때 초기값을 지정할 수 있음.  

```cpp
class Point {
  public:
      Point(int xVal, int yVal) {
        x = xVal; // 매개변수로 받은 x 좌표 값으로 초기화
        y = yVal; // 매개변수로 받은 y 좌표 값으로 초기화
      }
  private:
      int x; // x 좌표
      int y; // y 좌표
};
int main() {
    Point p(10, 20); // Point 객체 생성 시 매개변수가 있는 생성자가 호출되어 x가 10, y가 20으로 초기화됨
    p.print(); // Point(10, 20) 출력
    return 0;
}
```

- **초기화 리스트**: 생성자에서 정의시 멤버 변수 초기화를 : 뒤에 선언   

```cpp
class Point {
  public:
      Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 초기화 리스트를 사용하여 멤버 변수 초기화
      void print() {
          cout << "Point(" << x << ", " << y << ")" << endl; // 현재 좌표 출력
      }
  private:
      int x; // x 좌표
      int y; // y 좌표
};

int main() {
    Point p(10, 20); // Point 객체 생성 시 초기화 리스트를 사용하여 x가 10, y가 20으로 초기화됨
    p.print(); // Point(10, 20) 출력
    return 0;
}
```

- **복사 생성자**: 객체를 다른 객체로 초기화할 때 호출되는 생성자. 객체의 복사를 위해 사용됨.  

```cpp
class Point {
  public:
      Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 매개변수가 있는 생성자
      Point(const Point &p) : x(p.x), y(p.y) {} // 복사 생성자
      void print() {
          cout << "Point(" << x << ", " << y << ")" << endl; // 현재 좌표 출력
      }
  private:
      int x; // x 좌표
      int y; // y 좌표
};

int main() {
    Point p1(10, 20); // 매개변수가 있는 생성자를 사용하여 p1 객체 생성
    Point p2 = p1; // 복사 생성자를 사용하여 p2 객체 생성 (p1의 값으로 초기화)
    p2.print(); // Point(10, 20) 출력
    return 0;
}
```

- **Overloaded 생성자**: 동일한 클래스 내에서 여러 개의 생성자를 정의할 수 있으며, 매개변수의 개수나 타입에 따라 호출되는 생성자가 결정됨.   

```cpp
class Point {
  public:
      Point() : x(0), y(0) {} // 기본 생성자
      Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 매개변수가 있는 생성자
      Point(int xVal) : x(xVal), y(0) {} // x 좌표만 초기화하는 생성자
      void print() {
          cout << "Point(" << x << ", " << y << ")" << endl; // 현재 좌표 출력
      }
  private:
      int x; // x 좌표
      int y; // y 좌표
};

int main() {
    Point p1; // 기본 생성자 호출
    p1.print(); // Point(0, 0) 출력

    Point p2(10, 20); // 매개변수가 있는 생성자 호출
    p2.print(); // Point(10, 20) 출력

    Point p3(30); // x 좌표만 초기화하는 생성자 호출
    p3.print(); // Point(30, 0) 출력

    return 0;
}
```

- **주의사항**  
    - 생성자는 다른 멤버함수처럼 직접 호출 불가  
    - 다시 초기화하는 것은 대입 연산자를 사용해야 함   
    
```cpp
class Point {
  public:
      Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 매개변수가 있는 생성자
      Point(const Point &p) : x(p.x), y(p.y) {} // 복사 생성자
      void print() {
          cout << "Point(" << x << ", " << y << ")" << endl; // 현재 좌표 출력
      }
  private:
      int x; // x 좌표
      int y; // y 좌표
};

int main() {
    Point p1(10, 20); // 매개변수가 있는 생성자를 사용하여 p1 객체 생성
    p1.print(); // Point(10, 20) 출력

    // 생성자 호출 시도 (잘못된 예시)
    // Point p2 = Point(30, 40); // 컴파일 에러 발생: 생성자는 직접 호출할 수 없음

    // 기존 객체를 대입 연산자를 사용하여 다시 초기화는 가능
    p1 = Point(30, 40); // 대입 연산자를 사용하여 p1 객체를 새로운 값으로 초기화
    p1.print(); // Point(30, 40) 출력
    return 0;
}
```

## const
`const` 키워드는 다음과 같은 상황에서 사용될 수 있음

1. 변수 선언시 값이 변경되지 않도록 함
```cpp
const int x = 10; // x는 상수로, 이후 값 변경 불가
// x = 20; // 컴파일 에러 발생
```
2. 함수 매개변수로 전달되는 객체가 함수 내에서 변경되지 않도록 함
```cpp
Point::Point(const Point &p) : x(p.x), y(p.y) {} // const 참조로 매개변수 p를 받아서 복사 생성
```
- 해당 상황에서는 `const`를 사용하여 객체가 함수 내에서 변경되지 않도록 보장함.
- 여기서 Class 객체는 메모리가 크기 때문에, call-by-reference로 전달

3. 멤버 함수에서 객체의 상태를 변경하지 않음을 나타냄
```cpp
double Point::area() const { // const 멤버 함수
    return 3.14159 * radius * radius; // 객체의 상태를 변경하지 않음
}
```

## static  
`static` 키워드는 다음과 같은 상황에서 사용될 수 있음
1. **클래스 멤버 변수**: 클래스의 모든 객체가 공유하는 변수를 정의할 때 사용됨. 클래스의 모든 인스턴스가 동일한 값을 가지게 됨.
```cpp
class Point {
  public:
      static int count; // Point 개체 수를 저장하는 정적 멤버 변수
      Point() { count++; } // 생성자에서 count 증가
  private:
      int x; // x 좌표
      int y; // y 좌표
};
int Point::count = 0; // 정적 멤버 변수 초기화
int main() {
    Point p1; // Point 객체 생성
    Point p2; // 또 다른 Point 객체 생성
    cout << "Number of Point objects: " << Point::count << endl; // Point 객체 수 출력
    return 0;
}
```
- 위와 같이 모든 객체가 공유하는 변수가 필요할 때 `static` 키워드를 사용하여 클래스 멤버 변수를 정의할 수 있음.  
- 접근이 필요할 때는 `클래스이름::변수이름` 형식으로 접근함.
  - `객체이름.변수이름` 형식으로 접근할 수도 있지만, 권장되지 않음. 해당 값은 한 클래스 내 모든 객체가 공유하는 값이기 때문임.

1. **정적 멤버 함수**: 클래스의 인스턴스 없이 호출할 수 있는 함수를 정의할 때 사용됨. 정적 멤버 함수는 클래스의 정적 멤버 변수에만 접근할 수 있음.
```cpp
class Point {
  public:
      static int count; // Point 개체 수를 저장하는 정적 멤버 변수
      Point() { count++; } // 생성자에서 count 증가
      static int getCount() { return count; } // 정적 멤버 함수로 count 반환
  private:
      int x; // x 좌표
      int y; // y 좌표
};
int Point::count = 0; // 정적 멤버 변수 초기화
int main() {
    Point p1; // Point 객체 생성
    Point p2; // 또 다른 Point 객체 생성
    cout << "Number of Point objects: " << Point::getCount() << endl; // 정적 멤버 함수 호출
    return 0;
}
```
- 위와 같이 클래스의 인스턴스 없이 호출할 수 있는 함수를 정의할 때 `static` 키워드를 사용하여 정적 멤버 함수를 정의할 수 있음.
- `static` 멤버 함수는 클래스의 `static` 멤버 변수에만 접근할 수 있으며, 객체의 상태를 변경할 수 없음.  

## Vector(STL)
C++에서 **Vector**는 C++ 표준 템플릿 라이브러리(STL)의 일부로, 동적 배열을 구현한 컨테이너 클래스. Vector는 크기가 가변적이며, 요소를 추가하거나 제거할 수 있는 기능을 제공함. Vector는 배열과 유사하지만, 크기를 동적으로 조정할 수 있는 장점이 있음.
- **Vector 선언 및 초기화**:  
```cpp
#include <vector> // vector 헤더 파일 포함
using namespace std;
int main() {
    vector<int> vec; // 정수형 Vector 선언
    vector<string> strVec = {"Hello", "World"}; // 문자열형 Vector 선언 및 초기화

    // 요소 추가
    vec.push_back(1); // Vector에 1 추가
    vec.push_back(2); // Vector에 2 추가
    vec.push_back(3); // Vector에 3 추가

    // 요소 접근
    cout << "First element: " << vec[0] << endl; // 첫 번째 요소 출력
    cout << "Second element: " << vec.at(1) << endl; // 두 번째 요소 출력 (at() 함수 사용)

    // Vector 크기
    cout << "Vector size: " << vec.size() << endl; // Vector의 크기 출력

    return 0;
}
```

---
해당 포스트는 서울대학교 전기정보공학부 정교민 교수님의 프로그래밍방법론 25-1학기 강의를 정리한 내용입니다.
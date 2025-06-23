---
layout: post
title: 'Operator Overloading'
date: 2025-06-16 17:09 +0900
categories: ['C++']
tags: ['C++', '강의정리']
published: true
sitemap: true
math: true
---
## Operator Overloading
연산자(`+`, `-`, `*`, `/` 등)는 함수처럼 동작하며, 다만 다른 문법으로 호출되는 함수일뿐임.(예: `a + b`는 `operator+(a, b)`와 동일함)  
새로 정의한 Class에 대해서도 원하는대로 연산자를 사용할 수 있도록 하는 기능을 **연산자 오버로딩(Operator Overloading)** 이라고 함.

### 비멤버 함수 연산자 오버로딩
함수 오버로딩과 유사하게, 입력 타입과 반환 타입에 따라 연산자를 오버로딩할 수 있음.  
- **예시**: 두 개의 `Point` 객체를 더하는 연산자 오버로딩  

```cpp
class Point {
    public:
        Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 초기화 리스트 사용
        int getX() const { return x; } // x 좌표 반환
        int getY() const { return y; } // y 좌표 반환
        void setX(int xVal) { x = xVal; } // x 좌표 설정
        void setY(int yVal) { y = yVal; } // y 좌표 설정
    private:
        int x; // x 좌표
        int y; // y 좌표
};

const Point operator+(const Point& p1, const Point& p2) {
    return Point(p1.getX() + p2.getX(), p1.getY() + p2.getY());
}
```  
- 반환형이 `const Point`인 이유는, 연산 결과로 새로운 `Point` 객체를 반환할 때, 반환된 객체가 변경되지 않도록 하기 위함임.  
  - 예를 들어, `p1 + p2`와 같이 사용될 때, `(p1 + p2).setX(100)`와 같이 반환된 객체를 변경하는 것을 방지하기 위함

### 멤버 함수 연산자 오버로딩
멤버 함수로 연산자를 오버로딩할 수도 있음. 이 경우, 연산자 함수는 클래스의 멤버 변수에 직접 접근할 수 있음.  
- **예시**: `Point` 클래스의 `+` 연산자를 멤버 함수로 오버로딩  

```cpp
class Point {
    public:
        Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 초기화 리스트 사용
        int getX() const { return x; } // x 좌표 반환
        int getY() const { return y; } // y 좌표 반환
        void setX(int xVal) { x = xVal; } // x 좌표 설정
        void setY(int yVal) { y = yVal; } // y 좌표 설정

        // 멤버 함수로 연산자 오버로딩
        Point operator+(const Point& other) const;
    private:
        int x; // x 좌표
        int y; // y 좌표
};

Point Point::operator+(const Point& other) const {
    return Point(x + other.x, y + other.y);
}
```
### Friend 함수로 연산자 오버로딩
기본 연산자 오버로딩에서는, getter를 사용하여 멤버 변수에 접근하기 때문에 오버헤드가 발생할 수 있음.
`friend` 함수를 사용하면, 클래스의 private 멤버 변수에 직접 접근할 수 있어 성능을 향상시킬 수 있음. `friend` 함수는 클래스 외부에 정의되지만, 해당 클래스의 private 멤버 변수에 접근할 수 있는 권한을 가짐.
- **예시**: `Point` 클래스의 `+` 연산자를 `friend` 함수로 오버로딩  

```cpp
class Point {
    public:
        Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 초기화 리스트 사용
        int getX() const { return x; } // x 좌표 반환
        int getY() const { return y; } // y 좌표 반환
        void setX(int xVal) { x = xVal; } // x 좌표 설정
        void setY(int yVal) { y = yVal; } // y 좌표 설정

        // friend 함수로 연산자 오버로딩
        friend Point operator+(const Point& p1, const Point& p2);
    private:
        int x; // x 좌표
        int y; // y 좌표
};
Point operator+(const Point& p1, const Point& p2) {
    return Point(p1.x + p2.x, p1.y + p2.y);
}
```
`friend class F`와 같이 클래스 내부에 선언하면, 클래스 `F`의 모든 멤버 함수가 해당 클래스의 private 멤버 변수에 접근할 수 있게 됨.

```cpp
class Point {
    public:
        Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 초기화 리스트 사용
        int getX() const { return x; } // x 좌표 반환
        int getY() const { return y; } // y 좌표 반환
        void setX(int xVal) { x = xVal; } // x 좌표 설정
        void setY(int yVal) { y = yVal; } // y 좌표 설정

        friend class Circle; // Circle 클래스가 Point 클래스의 private 멤버에 접근할 수 있도록 허용
    private:
        int x; // x 좌표
        int y; // y 좌표
};
class Circle {
    public:
        Circle(int radiusVal, const Point& centerPoint) 
            : radius(radiusVal), center(centerPoint) {} // 초기화 리스트 사용
        bool isInside(const Point& p) const {
            // Circle의 중심점이 (0, 0)이라고 가정하고, Point p가 Circle 내부에 있는지 확인
            return ((p.x - center.x) * (p.x - center.x) + (p.y - center.y) * (p.y - center.y)) <= (radius * radius); // friend 선언으로 Point 클래스의 private 멤버에 접근 가능
        }
    private:
        int radius; // 반지름
        Point center; // 원의 중심점
};
```

### 다양한 연산자 오버로딩
- `()` 연산자: 함수 호출 연산자를 오버로딩하여 객체를 함수처럼 사용할 수 있음.
  - 오직 멤버 함수로만 오버로딩 가능함.  

```cpp
class Point {
    public:
        Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 초기화 리스트 사용
        int getX() const { return x; } // x 좌표 반환
        int getY() const { return y; } // y 좌표 반환
        void setX(int xVal) { x = xVal; } // x 좌표 설정
        void setY(int yVal) { y = yVal; } // y 좌표 설정

        // 함수 호출 연산자 오버로딩
        int operator()(int multiplier) const {
            return (x + y) * multiplier; // (x + y)에 multiplier를 곱한 값을 반환
        }
    private:
        int x; // x 좌표
        int y; // y 좌표
};

int main() {
    Point p(3, 4);
    int result = p(2); // p(2)는 (3 + 4) * 2 = 14를 반환
    std::cout << "Result: " << result << std::endl; // Result: 14 출력
    return 0;
}
```

- **Unary 연산자**: 단항 연산자를 오버로딩하여 객체에 대해 단일 연산을 수행할 수 있음.
  - 예: `++`, `--`, `!`, `-` 등  

```cpp
#include <iostream>

class Point {
public:
    Point(int xVal, int yVal) : x(xVal), y(yVal) {}

    int getX() const { return x; }
    int getY() const { return y; }
    void setX(int xVal) { x = xVal; }
    void setY(int yVal) { y = yVal; }

    // 단항 - 연산자
    Point operator-() const {
        return Point(-x, -y);
    }

    // 이항 - 연산자
    Point operator-(const Point& other) const {
        return Point(x - other.x, y - other.y);
    }

    // 전위 ++ 연산자
    Point& operator++() {
        ++x;
        ++y;
        return *this;
    }

private:
    int x;
    int y;
};

int main() {
    Point p(3, 4);
    ++p; // p의 x와 y를 각각 1씩 증가시킴
    std::cout << "After increment: (" << p.getX() << ", " << p.getY() << ")" << std::endl; // After increment: (4, 5) 출력
    return 0;
}
```

- `<<`, `>>` 연산자: 입출력 스트림에 대한 연산자를 오버로딩하여 객체를 출력하거나 입력할 수 있음.
  - `<<`는 출력 스트림(`std::ostream`)에 객체를 출력하는 데 사용되고, `>>`는 입력 스트림(`std::istream`)에서 객체를 입력받는 데 사용됨.  


```cpp
// 출력 스트림 연산자 오버로딩
friend ostream& operator<<(ostream& os, const Point& p) {
    os << "(" << p.x << ", " << p.y << ")"; // (x, y) 형식으로 출력
    return os;
}

// 입력 스트림 연산자 오버로딩
friend istream& operator>>(istream& is, Point& p) {
    char ch; // '('와 ','를 읽기 위한 변수
    is >> ch >> p.x >> ch >> p.y >> ch; // (x, y) 형식으로 입력받음
    return is;
}

int main() {
    Point p(3, 4);
    cout << "Point: " << p << endl; // Point: (3, 4) 출력

    cout << "Enter a point in the format (x, y): ";
    cin >> p; // 사용자로부터 입력받음
    cout << "Updated Point: " << p << endl; // Updated Point: (입력된 x, 입력된 y) 출력

    return 0;
}
```
- 여기서 출력 스트림의 reference `ostream&`과 입력 스트림의 reference `istream&`을 사용하는 이유는, `ostream`객체가 그냥 복사되면, 출력 스트림 연결이 끊어지기 때문임.

- `=` 연산자: 대입 연산자를 오버로딩하여 객체를 다른 객체로 대입할 수 있음.
  - 이 연산자는 반드시 멤버 함수로 오버로딩해야 함.
  - 단순 클래스는 기본 대입 연산자가 제공되지만, 포인터나 동적 메모리를 사용하는 클래스에서는 직접 구현해야 함.(Shallow Copy 방지)  

```cpp
// 대입 연산자 오버로딩
Point& operator=(const Point& other) {
    if (this != &other) { // 자기 자신과의 대입 방지
        x = other.x; // x 좌표 복사
        y = other.y; // y 좌표 복사
    }
    return *this; // 자기 자신을 반환
}
```

- `++`, `--` 연산자: 증가 및 감소 연산자를 오버로딩하여 객체의 값을 증가시키거나 감소시킬 수 있음.
  - 이 연산자는 멤버 함수로 오버로딩할 수 있으며, 전위(`++p`)와 후위(`p++`) 연산자 dummy 인자를 사용하여 구분할 수 있음.(`operator++(int)`와 같이 int형 dummy 인자를 받는 후위 연산자)  

```cpp
Point& operator++() { // 전위 증가 연산자
    ++x; // x 좌표 증가
    ++y; // y 좌표 증가
    return *this; // 자기 자신을 반환
}
Point operator++(int) { // 후위 증가 연산자
    Point temp = *this; // 현재 객체를 임시 변수에 저장
    ++x; // x 좌표 증가
    ++y; // y 좌표 증가
    return temp; // 증가 전의 객체를 반환
}
```
### Automatic Type Conversion
만약 한 자료형에서 클래스로의 자동 형변환(생성자)가 정의되어 있다면, 연산자 오버로딩을 사용할 때 자동으로 형변환이 일어날 수 있음.  
- **예시**: `int`를 `Point`로 자동 변환하여 더하기 연산을 수행하는 경우  

```cpp
class Point {
    public:
        Point(int xVal, int yVal) : x(xVal), y(yVal) {} // 초기화 리스트 사용
        int getX() const { return x; } // x 좌표 반환
        int getY() const { return y; } // y 좌표 반환
        void setX(int xVal) { x = xVal; } // x 좌표 설정
        void setY(int yVal) { y = yVal; } // y 좌표 설정

        // int를 Point로 변환하는 생성자
        Point(int value) : x(value), y(value) {}

        // + 연산자 오버로딩
        Point operator+(const Point& other) const {
            return Point(x + other.x, y + other.y);
        }
    private:
        int x; // x 좌표
        int y; // y 좌표
};

int main() {
    int value = 5;
    Point p1(3, 4);
    Point p2 = p1 + value; // p1 + int value, 자동 형변환이 일어남
    std::cout << "Result: (" << p2.getX() << ", " << p2.getY() << ")" << std::endl; // Result: (8, 8) 출력
    return 0;
}
```

이와 같은 Automatic Type Conversion은 멤버 함수 오버로딩에서는 우항에 있는 객체에 대해서만 일어남
- `Point Point::operator+(const Point& other) const`와 같이 멤버 함수로 오버로딩된 경우, `int+ Point`와 같은 연산은 자동 형변환이 일어나지 않음.
- `Point operator+(const Point& p1, const Point& p2)`와 같이 비멤버 함수로 오버로딩된 경우, `int + Point`와 같은 연산에서 자동 형변환이 일어남.
- `frient Point operator+(const Point& p1, const Point& p2)`도 마찬가지
  
### 연산자 오버로딩 방식의 비교  


| 연산자 오버로딩 방식 | 장점                                                                                                       | 단점                                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 멤버 함수            | - 클래스의 멤버 변수에 직접 접근 가능<br>- 객체의 상태를 변경할 수 있음                                    | - 연산자 우항에 있는 객체에 대해서는 자동 형변환이 일어나지 않음                                              |
| 비멤버 함수          | - 연산자 우항에 있는 객체에 대해서도 자동 형변환이 일어남<br>- 클래스 외부에서 정의 가능                   | - 클래스의 private 멤버 변수에 접근할 수 없음<br>- 멤버 변수에 접근하려면 getter를 사용해야 함(오버헤드 발생) |
| friend 함수          | - 클래스의 private 멤버 변수에 직접 접근 가능<br>- 연산자 우항에 있는 객체에 대해서도 자동 형변환이 일어남 | - 객체 지향 원칙에 위배된다는 비판이 있음                                                                     |

---
해당 포스트는 서울대학교 전기정보공학부 정교민 교수님의 프로그래밍방법론 25-1학기 강의를 정리한 내용입니다.
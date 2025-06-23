---
layout: post
title: 'Structure & Classes'
date: 2025-06-16 11:08 +0900
categories: ['C++']
tags: ['C++', '강의정리']
published: true
sitemap: true
math: true
---
## Structure
Structure는 C++에서 데이터를 그룹화하는 데 사용되는 사용자 정의 데이터 타입. Array와 유사하지만, Structure는 서로 다른 데이터 타입을 포함할 수 있음.   
Structure는 `struct` 키워드를 사용하여 정의되며 이 구조체에 저장된 데이터는 멤버 변수, 함수는 멤버 함수로 불림. 
- Structure 정의 예시:  
```cpp
struct Point {
    int x; // x 좌표
    int y; // y 좌표

    // 멤버 함수
    void move(int dx, int dy) {
        x += dx;
        y += dy;
    }
}; //struct로 정의할 때, 반드시 `;`로 끝나야 함.
```  
- Structure 변수 선언 및 초기화:
```cpp
Point p1; // 구조체 변수 선언
// 초기화의 두 가지 방법
// 멤버 변수에 직접 접근하여 초기화
p1.x = 10; // x 좌표 초기화
p1.y = 20; // y 좌표 초기화
// 선언과 동시에 {}를 사용하여 초기화
Point p2 = {30, 40}; // 구조체 변수 초기화
```
- Structure는 일반 데이터 타입처럼 argument(Call-by-ref, Call-by-value 둘 다 가능), return type으로 사용 가능하며, 같은 type의 Structure끼리 assignment도 가능함.  
  
```cpp
void printPoint(Point p) {
    std::cout << "Point(" << p.x << ", " << p.y << ")" << std::endl;
}
Point p3 = {50, 60};
printPoint(p3); // Point(50, 60) 출력
Point p4 = p2; // p2의 값을 p4에 복사
Point doublePoint(Point p) {
    p.x *= 2; // x 좌표를 두 배로
    p.y *= 2; // y 좌표를 두 배로
    return p; // 변경된 구조체 반환
}
```

## Class 
Class는 C++에서 객체 지향 프로그래밍(OOP)을 지원하는 사용자 정의 데이터 타입. Structure와 기본적으로 같은 개념이지만, Class는 기본적으로 멤버 변수와 멤버 함수를 지정했을때 private로 설정됨.  
Class는 `class` 키워드를 사용하여 정의되며, 멤버 변수와 멤버 함수는 public, private, protected 접근 제어자를 통해 접근 권한을 설정할 수 있음.  
- Class 정의 예시:  
```cpp
class Circle {
private:
    double radius; // 반지름
public:
    // 생성자
    Circle(double r) : radius(r) {}

    // 멤버 함수
    double area() const { // const는 이 함수가 객체의 상태를 변경하지 않음을 나타냄
        return 3.14159 * radius * radius; // 원의 면적 계산
    }

    void setRadius(double r) { // 반지름 설정 함수
        radius = r;
    }
};
```  
위 예시에서, `Circle` 클래스의 `radius` 멤버 변수는 private로 설정되어 structure와 달리 외부에서 직접 접근할 수 없음. 대신, `setRadius`라는 public 멤버 함수를 통해 반지름을 설정할 수 있음.  
- Class 변수 선언 및 초기화:
```cpp
Circle c1(5.0); // Circle 객체 생성 및 초기화
Circle c2(10.0); // Circle 객체 생성 및 초기화
// 면적 계산
double area1 = c1.area(); // c1의 면적 계산
double area2 = c2.area(); // c2의 면적 계산
std::cout << "Area of c1: " << area1 << std::endl; // c1의 면적 출력
std::cout << "Area of c2: " << area2 << std::endl; // c2의 면적 출력
// 반지름 변경
c1.setRadius(7.0); // c1의 반지름을 7.0으로 변경
c2.radius = 12.0; // compile error: radius는 private이므로 직접 접근 불가
```
멤버 함수는 정의만 해두고, 클래스 외부에서 구현할 수 있음.  

```cpp
class Rectangle {
private:
    double width; // 너비
    double height; // 높이
public:
    Rectangle(double w, double h) : width(w), height(h) {} // 생성자

    double area() const; // 면적 계산 함수 선언
};

double Rectangle::area() const { // 클래스 외부에서 구현
    return width * height; // 직사각형의 면적 계산
}
```

## 객체 지향 프로그래밍(OOP)의 원리
객체 지향 프로그래밍(OOP)은 프로그램을 객체 단위로 구성하여 코드의 재사용성과 유지보수성을 높이는 프로그래밍 패러다임. C++에서 OOP의 주요 원리는 다음과 같음:  
- **캡슐화(Encapsulation)**: 객체의 상태(데이터)와 행동(함수)을 하나의 단위로 묶어 외부에서 직접 접근하지 못하도록 보호하는 것. 이를 통해 데이터의 무결성을 유지하고, 객체의 내부 구현을 숨길 수 있음.  
```cpp
class BankAccount {
private:
    double balance; // 잔액
public:
    BankAccount(double initialBalance) : balance(initialBalance) {}

    void deposit(double amount) { // 입금 함수
        if (amount > 0) {
            balance += amount;
        }
    }

    void withdraw(double amount) { // 출금 함수
        if (amount > 0 && amount <= balance) {
            balance -= amount;
        }
    }

    double getBalance() const { // 잔액 조회 함수
        return balance;
    }
};
```
- **상속(Inheritance)**: 기존 클래스(부모 클래스)의 속성과 행동을 새로운 클래스(자식 클래스)가 물려받아 재사용하는 것. 이를 통해 코드의 중복을 줄이고, 계층 구조를 형성할 수 있음.  
```cpp
class Person {
private:
    std::string name; // 이름
    int age; // 나이
public:
    Person(std::string n, int a) : name(n), age(a) {}

    void introduce() const {
        std::cout << "My name is " << name << " and I am " << age << " years old." << std::endl;
    }
};
class Student : public Person { // Person 클래스를 상속받음
private:
    std::string major; // 전공
public:
    Student(std::string n, int a, std::string m) : Person(n, a), major(m) {}

    void study() const {
        std::cout << "I am studying " << major << "." << std::endl;
    }
};
```
위 예시에서 `Student` 클래스는 `Person` 클래스를 상속받아 이름과 나이를 그대로 사용할 수 있으며, 추가로 전공을 나타내는 `major` 멤버 변수를 가짐.
- **데이터 추상화(Data Abstraction)**: 객체의 복잡한 내부 구현을 숨기고, 필요한 정보만 외부에 제공하는 것. 이를 통해 사용자는 객체의 내부 구조를 알 필요 없이 인터페이스를 통해 객체와 상호작용할 수 있음.  
```cpp
class User {
private:
    std::string username; // 사용자 이름
    std::string password; // 비밀번호
public:
    User(std::string u, std::string p) : username(u), password(p) {}

    bool login(std::string inputPassword) { // 로그인 함수
        return inputPassword == password; // 비밀번호가 일치하면 true 반환
    }

    void displayUsername() const { // 사용자 이름 출력 함수
        std::cout << "Username: " << username << std::endl;
    }
};
```
위 예시에서 `User` 클래스는 사용자 이름과 비밀번호를 private로 설정하여 외부에서 직접 접근할 수 없도록 하고, 로그인 기능을 제공하는 `login` 함수를 통해 비밀번호 검증을 수행함. 사용자는 객체의 내부 구조를 알 필요 없이 `login` 함수와 `displayUsername` 함수를 통해 상호작용할 수 있음.

### Class와 Structure에서의 상속
C++에서 Class와 Structure는 기본적으로 동일한 기능을 제공하지만, 접근 제어의 기본값이 다름. Class는 기본적으로 private 멤버로 설정되며, Structure는 public 멤버로 설정됨.  
이는 상속에서도 동일하게 적용됨. Class와 Structure 모두 상속을 지원하지만, 상속의 접근 제어를 설정하지 않으면 Class는 private 상속, Structure는 public 상속이 기본값임.  
```cpp
class Animal {
    public:
        void eat() {
            std::cout << "Eating..." << std::endl;
        }
};

struct StructDog : Animal { // Structure로 기본 상속
    void bark() {
        std::cout << "Barking..." << std::endl;
    }
};

class ClassDog : Animal { // Class로 기본 상속
    public:
        void bark() {
            std::cout << "Barking..." << std::endl;
        }
};

int main() {
    StructDog dog1; // Structure로 생성
    dog1.eat(); // 상속받은 eat 함수 호출
    dog1.bark(); // Structure의 bark 함수 호출

    ClassDog dog2; // Class로 생성
    dog2.eat(); // 상속이 private이므로 ClassDog에서 eat 함수는 접근 불가, compile error 발생
    dog2.bark(); // Class의 bark 함수 호출

    return 0;
}
```
- 상속 지정자: C++에서 상속을 지정할 때는 `public`, `protected`, `private` 키워드를 사용하여 상속의 접근 제어를 설정할 수 있음.   

| 부모 멤버 | public 상속          | protected 상속       | private 상속         |
| --------- | -------------------- | -------------------- | -------------------- |
| public    | 그대로 public        | protected로          | private로            |
| protected | 그대로 protected     | 그대로 protected     | private로            |
| private   | 자식에서 직접 접근 ❌ | 자식에서 직접 접근 ❌ | 자식에서 직접 접근 ❌ |


## Accessor & Mutator
Accessor와 Mutator는 객체 지향 프로그래밍에서 객체의 상태를 안전하게 접근하고 수정하기 위한 메서드. Accessor는 객체의 상태를 읽기 위한 메서드, Mutator는 객체의 상태를 변경하기 위한 메서드로 사용됨.
- **Accessor**: 객체의 멤버 변수 값을 반환하는 메서드. 일반적으로 `get` 접두사를 사용하여 정의됨.  
```cpp
class Person {
private:
    std::string name; // 이름
    int age; // 나이
public:
    Person(std::string n, int a) : name(n), age(a) {}

    // Accessor 메서드
    std::string getName() const { // 이름 반환
        return name;
    }

    int getAge() const { // 나이 반환
        return age;
    }
};
```
- **Mutator**: 객체의 멤버 변수 값을 변경하는 메서드. 일반적으로 `set` 접두사를 사용하여 정의됨.  
```cpp
class Person {
private:
    std::string name; // 이름
    int age; // 나이
public:
    Person(std::string n, int a) : name(n), age(a) {}

    // Mutator 메서드
    void setName(const std::string& n) { // 이름 설정
        name = n;
    }

    void setAge(int a) { // 나이 설정
        if (a >= 0) { // 유효성 검사
            age = a;
        }
    }
};
```
Accessor와 Mutator를 사용하면 객체의 상태를 직접 변경하거나 읽는 것을 방지하고, 필요한 경우 유효성 검사나 추가 로직을 구현할 수 있음. 이를 통해 객체의 무결성을 유지하고, 코드의 가독성과 유지보수성을 향상시킬 수 있음.  

---
해당 포스트는 서울대학교 전기정보공학부 정교민 교수님의 프로그래밍방법론 25-1학기 강의를 정리한 내용입니다.
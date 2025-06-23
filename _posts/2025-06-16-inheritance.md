---
layout: post
title: 'Inheritance'
date: 2025-06-16 19:43 +0900
categories: ['C++']
tags: ['C++', '강의정리']
published: true
sitemap: true
math: true
---
## Inheritance
상속(Inheritance)은 객체 지향 프로그래밍의 핵심 개념 중 하나로, 기존 클래스(부모 클래스)의 속성과 메서드를 새로운 클래스(자식 클래스)가 물려받는 기능을 의미함. 이를 통해 코드의 재사용성을 높이고, 계층 구조를 형성할 수 있음.  
기본 클래스(Base Class)로부터 파생된 클래스(Derived Class)는 기본 클래스의 멤버 변수와 메서드를 자동으로 상속받음. 자식 클래스는 부모 클래스의 기능을 확장하거나 수정할 수 있으며, 새로운 멤버 변수와 메서드를 추가할 수 있음.

### 예시

```cpp
class Employee {
    public:
        Employee(string name, string ssn) : name(name), ssn(ssn) {}
        void setName(string newName) { name = newName; }
        string getName() const { return name; }
        virtual void printcheck();
    protected:
        string name; 
        string ssn;  
        double pay;
};

class HourlyEmployee : public Employee {
private:
    double wageRate;
    double hoursWorked;
public:
    HourlyEmployee(string name, string ssn, double wage, double hours);
    void setRate(double wage);
    double getRate();
    void setHours(double hours);
    double getHours();
    void printCheck() override;  // printCheck 재정의
};
```
- `printCheck()`  
    기본 클래스에서 무의미한 함수로 정의, 자식 클래스에서 반드시 재정의해야 함.
- 새로운 멤버 변수: `wageRate`, `hoursWorked`  
    자식 클래스에서만 사용하는 멤버 변수로, 부모 클래스의 멤버 변수와는 별개로 정의됨.
- 새로운 메서드: `setRate()`, `getRate()`, `setHours()`, `getHours()`  
    자식 클래스에서만 사용하는 메서드로, 부모 클래스의 메서드와는 별개로 정의됨.

### Constructors & Destructors
부모 클래스의 생성자는 자식 클래스의 생성자에서 자동으로 호출되고, 상속되는 것이 아님. 자식 클래스의 생성자에서 부모 클래스의 생성자를 명시적으로 호출해야 함.  
호출하지 않는 경우, 부모 클래스의 기본 생성자가 자동으로 호출됨.  

```cpp
HourlyEmployee::HourlyEmployee(string n, string s, double w, double h) 
: Employee(n, s), wageRate(w), hoursWorked(h) {}
```
- `Employee(n, s)`는 부모 클래스의 생성자를 호출하는 부분으로, 부모 클래스의 멤버 변수를 초기화함.
- `wageRate(w)`와 `hoursWorked(h)`는 자식 클래스의 멤버 변수를 초기화하는 부분임.
- destructor가 실행될때는, 파생 클래스의 destructor가 먼저 실행되고, 그 다음에 기본 클래스의 destructor가 실행됨.  
  - 이는 자식 클래스의 자원 해제를 먼저 하고, 부모 클래스의 자원을 해제하기 위함임.

### Access Specifiers
상속 시, 부모 클래스의 멤버 변수와 메서드에 대한 접근 권한을 지정할 수 있음. private 멤버는 메모리상 상속은 되지만, 자식 클래스에서는 접근할 수 없다. 
- **public**: 부모 클래스의 public 멤버는 자식 클래스에서도 public으로, protected는 protected로 상속됨.
- **protected**: 부모 클래스의 public 멤버는 자식 클래스에서 protected로, protected는 protected로 상속됨.
- **private**: 부모 클래스의 public 멤버는 자식 클래스에서 private로, protected는 private로 상속됨.

```cpp
class Base {
    public:
        int publicVar; // public 멤버
    protected:
        int protectedVar; // protected 멤버
    private:
        int privateVar; // private 멤버
};
class Derived : public Base {
    public:
        void accessMembers() {
            publicVar = 1; // 접근 가능
            protectedVar = 2; // 접근 가능
            // privateVar = 3; // 접근 불가, 컴파일 에러 발생
        }
};
```

### Function Redefinition vs Overriding
- **Function Redefinition**: 부모 클래스의 멤버 함수를 자식 클래스에서 같은 이름으로 정의하는 것. 이 경우, 부모 클래스의 함수는 숨겨지고, 자식 클래스의 함수가 호출됨.  
- **Overriding**: 부모 클래스의 멤버 함수를 자식 클래스에서 재정의하는 것. 이 경우, 부모 클래스의 함수는 virtual로 선언되어 있어야 하며, 자식 클래스에서 같은 이름과 매개변수를 가진 함수를 정의함.  
  - 이때, 부모 클래스의 함수는 자식 클래스에서 재정의된 함수가 호출됨.

```cpp
class Base {
    public:
        virtual void display() { // virtual로 선언된 함수
            std::cout << "Base class display function" << std::endl;
        }
        void add(int a, int b) { // 일반 함수
            std::cout << "Sum: " << (a + b) << std::endl;
        }
};
class Derived : public Base {
    public:
        void display() override { // 부모 클래스의 함수를 재정의
            std::cout << "Derived class display function" << std::endl;
        }
        void add(int a, int b, int c) { // 새로운 매개변수를 가진 함수(Function Overloading)
            std::cout << "Sum with three parameters: " << (a + b + c) << std::endl;
        }
};
```
- 오버라이딩은 함수명+매개변수의 타입과 개수로 구성된 함수 시그니처가 동일해야 함.
- Redefinition은 부모 클래스의 함수와 매개변수가 동일하지 않아도 됨.
- Redefinition은 부모 클래스의 함수를 기본적으로 숨기지만, 숨겨진 부모 클래스의 함수를 호출할 수 있음.  
```cpp
HourlyEmployee he("John Doe", "123-45-6789", 20.0, 40.0);
he.printCheck(); // 자식 클래스의 printCheck 호출
he.Employee::printCheck(); // 부모 클래스의 printCheck 호출
```

### Functions not inherited
- 상속관계에 있어도 상속되지 않는 함수들이 있음:  
  - **Constructors**
  - **Destructors**
  - **Copy Constructors**
  - **Assignment Operators**

#### Copy Constructors
복사 생성자는 객체를 복사할 때 호출되는 특별한 생성자임. 상속 관계에서는 부모 클래스의 복사 생성자가 자식 클래스의 복사 생성자에 의해 호출되지 않음.  
따라서, 자식 클래스에서 복사 생성자를 정의할 때, 부모 클래스의 복사 생성자를 명시적으로 호출해야 함.  
```cpp
class Base {
    public:
        Base(int value) : data(value) {}
        Base(const Base& other) : data(other.data) {} // 복사 생성자
    private:
        int data;
};
class Derived : public Base {
    public:
        Derived(int value, int extra) : Base(value), extraData(extra) {}
        Derived(const Derived& other) : Base(other), extraData(other.extraData) {} // 복사 생성자
    private:
        int extraData;
};
int main() {
    Derived d1(10, 20);
    Derived d2 = d1; // 복사 생성자 호출
    return 0;
}
```

#### Assignment Operators
대입 연산자는 객체를 대입할 때 호출되는 특별한 함수. 상속 관계에서는 부모 클래스의 대입 연산자가 자식 클래스의 대입 연산자에 의해 호출되지 않음.  
따라서, 자식 클래스에서 대입 연산자를 정의할 때, 부모 클래스의 대입 연산자를 명시적으로 호출해야 함.  
```cpp
class Base {
    public:
        Base(int value) : data(value) {}
        Base& operator=(const Base& other) { // 대입 연산자
            if (this != &other) {
                data = other.data;
            }
            return *this;
        }
    private:
        int data;
};
class Derived : public Base {
    public:
        Derived(int value, int extra) : Base(value), extraData(extra) {}
        Derived& operator=(const Derived& other) { // 대입 연산자
            if (this != &other) {
                Base::operator=(other); // 부모 클래스의 대입 연산자 호출
                extraData = other.extraData;
            }
            return *this;
        }
    private:
        int extraData;
};
int main() {
    Derived d1(10, 20);
    Derived d2(30, 40);
    d2 = d1; // 대입 연산자 호출
    return 0;
}
```

### Multiple Inheritance
C++는 다중 상속(Multiple Inheritance)을 지원함. 즉, 하나의 자식 클래스가 여러 부모 클래스로부터 상속받을 수 있음.  
하지만, 다중 상속은 복잡성을 증가시키고, 다이아몬드 문제(Diamond Problem)와 같은 문제를 발생시킬 수 있음.  
다이아몬드 문제는 두 부모 클래스가 동일한 부모 클래스를 상속받을 때 발생함. 이 경우, 자식 클래스가 부모 클래스의 멤버를 중복으로 상속받게 되어 모호성이 발생할 수 있음.  
```cpp
class A {
    public:
        void display() { std::cout << "Class A" << std::endl; }
};
class B : public A {
    public:
        void display() { std::cout << "Class B" << std::endl; }
};
class C : public A {
    public:
        void display() { std::cout << "Class C" << std::endl; }
};
class D : public B, public C {
    // D는 B와 C를 상속받음
};
int main() {
    D d;
    // d.display(); // 컴파일 에러: B와 C 모두 A를 상속받아 모호함
    d.B::display(); // B의 display 호출
    d.C::display(); // C의 display 호출
    return 0;
}
```

---
해당 포스트는 서울대학교 전기정보공학부 정교민 교수님의 프로그래밍방법론 25-1학기 강의를 정리한 내용입니다.
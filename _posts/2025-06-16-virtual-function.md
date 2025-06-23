---
layout: post
title: 'Virtual Function'
date: 2025-06-16 20:25 +0900
categories: ['C++']
tags: ['C++', '강의정리']
published: true
sitemap: true
math: true
---
## Virtual Function(Polymorphism)
C++에서 **가상 함수(Virtual Function)** 는 **다형성(Polymorphism)** 을 구현하는 데 사용되는 특별한 멤버 함수로, 상속 관계에 있는 클래스에서 오버라이딩할 수 있는 함수를 의미함. 가상 함수는 런타임에 동적으로 바인딩되어, 객체의 실제 타입에 따라 호출되는 함수가 결정됨.
- 다형성은 동일한 인터페이스를 사용하여 서로 다른 객체를 처리할 수 있는 능력을 의미함.
- virtual function이 이를 지원함
- "virtual"이란 뜻은 함수의 정의가 명확하지 않고, 실제 런타임에 결정된다는 의미임.(Late Binding)
- 예:   

```cpp
class Figure {
public:
    virtual void draw() = 0;  // 순수 가상 함수 → abstract class
    void center() {
        cout << "Erase\n";
        cout << "Move to center\n";
        draw();  // 실제 객체 타입의 draw() 호출
    }
};

class Rectangle : public Figure {
public:
    void draw() override { cout << "Rectangle::draw()\n"; }
};

class Circle : public Figure {
public:
    void draw() override { cout << "Circle::draw()\n"; }
};
```
- 위 예시에서 `Figure` 클래스는 순수 가상 함수 `draw()`를 가지고 있어, 이 클래스를 상속받는 클래스는 반드시 `draw()` 함수를 구현해야 함.
- `center()` 함수는 `draw()` 함수를 호출하는데, 이때 실제 객체의 타입에 따라 적절한 `draw()` 함수가 호출됨.
- 만약 virtual 키워드가 없었다면, `draw()` 함수는 컴파일 타임에 결정되어, `Figure` 클래스의 `draw()` 함수가 호출되었을 것임.

```cpp
class Sale {
public:
    Sale(double price = 0.0) : price(price) {}
    virtual double bill() const { return price; }
    double savings(const Sale& other) const { return bill() - other.bill(); }
    double getPrice() const { return price; }
private:
    double price;
};

bool operator<(const Sale& s1, const Sale& s2) {
    return s1.bill() < s2.bill();
}

class DiscountSale : public Sale {
public:
    DiscountSale(double price = 0.0, double discount = 0.0)
        : Sale(price), discount(discount) {}
    double bill() const override {
        return (1 - discount / 100) * getPrice();
    }
private:
    double discount;
};

int main() {
    Sale s1(10.0);
    DiscountSale s2(11.0, 10.0);

    if (s2 < s1) {
        cout << "Savings: " << s1.savings(s2) << endl;
    } 
    ...
    return 0;
}
```
- 위 예시에서, `s2 < s1` 비교는 `DiscountSale` 클래스의 `bill()` 과 `Sale` 클래스의 `bill()` 함수를 호출하여, 각각의 객체에 맞는 가격을 계산함.
- `DiscountSale` 클래스가 정의되기 전에 `<` 연산자가 오버로딩 되어있지만, 런타임에서 알아서 적절한 `bill()` 함수를 호출함.

### Redefine과의 비교  

```cpp
class Figure {
public:
    void draw() { 
        cout << "Figure::draw()\n"; 
    }

    void center() { 
        draw(); 
    }
};

class Circle : public Figure {
public:
    void draw() { 
        cout << "Circle::draw()\n"; 
    }
};
int main() {
    Circle c;
    c.center();
    return 0;
}
```
- 위와 같은 식으로 `draw()` 함수를 재정의할때, `c`의 `center()` 함수는 `Figure` 클래스의 `draw()` 함수를 호출함.

### Pure Virtual Function & Abstract Class
- **순수 가상 함수(Pure Virtual Function)** 는 클래스가 반드시 오버라이드해야 하는 함수를 정의할 때 사용됨. 순수 가상 함수는 `= 0`으로 선언되어, 해당 클래스를 추상 클래스(Abstract Class)로 만듦.
- 추상 클래스는 인스턴스를 생성할 수 없으며, 오직 상속을 통해서만 사용될 수 있음.
- 예시:  

```cpp
class AbstractShape {
public:
    virtual void draw() = 0;  // 순수 가상 함수
    virtual double area() = 0; // 순수 가상 함수
};
class Circle : public AbstractShape {
public:
    Circle(double r) : radius(r) {}
    void draw() override {
        cout << "Drawing Circle with radius: " << radius << endl;
    }
    double area() override {
        return 3.14 * radius * radius;
    }
private:
    double radius;
};
int main() {
    Circle c(5.0);
    c.draw();  // "Drawing Circle with radius: 5"
    cout << "Area: " << c.area() << endl;  // "Area: 78.5"
    // AbstractShape s; // 컴파일 에러: 추상 클래스는 인스턴스화할 수 없음
    return 0;
}
```
- 위 예시에서 `AbstractShape` 클래스는 순수 가상 함수 `draw()`와 `area()`를 가지고 있어, 이 클래스를 상속받는 클래스는 반드시 이 함수를 구현해야 함.
  - 구현하지 않으면, 해당 클래스도 추상 클래스가 되어 인스턴스를 생성할 수 없음.

### Pointer(Upcasting & Downcasting)
- **Upcasting**: 자식 클래스의 객체를 부모 클래스의 포인터로 변환하는 것. 이는 안전하며, 부모 클래스의 멤버 함수만 호출할 수 있음.
- **Downcasting**: 부모 클래스의 포인터를 자식 클래스의 포인터로 변환하는 것. 이는 위험할 수 있음. 
- **Slicing**: 자식 클래스의 객체를 부모 클래스의 객체로 변환할 때, 자식 클래스의 멤버가 잘려나가는 현상.

- 예시:  

```cpp
class Dog {
public:
    virtual void bark() {
        cout << "Dog barks" << endl;
    }
};
class Retriever : public Dog {
public:
    void bark() override {
        cout << "Retriever barks" << endl;
    }
    void fetch() {
        cout << "Retriever fetches" << endl;
    }
};
int main() {
    Dog* dogPtr = new Retriever();  // Upcasting
    dogPtr->bark();  // "Retriever barks"

    // Retriever* retrieverPtr = new Dog();  컴파일 에러: Dog 객체를 Retriever로 변환할 수 없음

    Retriever r;
    Dog d = r;  // Slicing 발생: Retriever 객체의 멤버가 잘려나감
}
```
- 위 예시에서 `Dog` 클래스의 포인터로 `Retriever` 클래스의 객체를 받는 것은 Upcasting이며, 이는 안전하게 수행됨.
- 반면, `Dog` 클래스의 포인터를 `Retriever` 클래스의 포인터로 변환하는 것은 Downcasting이며, 이는 위험할 수 있음. 만약 `Dog` 객체가 `Retriever` 객체가 아니라면, `fetch()` 함수를 호출하려고 할 때 런타임 에러가 발생할 수 있음.

- **Type Casting of Classes**
  - 클래스 간 타입캐스팅이 지원되기도 함
  - 이 경우에도 역시, Upcasting은 안전하지만, Downcasting은 위험할 수 있음.
  - 예시:  

```cpp
Dog dog;
Retriever retriever;
// retriever = static_cast<Retriever>(dog); Downcasting, ILLEGAL
dog = retriever;  // Slicing 발생, 하지만 LEGAL
dog = static_cast<Dog>(retriever);  // Upcasting, LEGAL
Dog* dogPtr;
dogPtr = new Retriever(); 
Retriever* retrieverPtr = dynamic_cast<Retriever*>(dogPtr);  // Downcasting, LEGAL, 만약 dogPtr가 Retriever 객체가 아니라면, retrieverPtr는 nullptr이 됨
```


### Virtual Destructor
- **가상 소멸자(Virtual Destructor)** 는 상속 관계에 있는 클래스에서 객체가 삭제될 때, 올바른 소멸자가 호출되도록 보장하는 데 사용됨. 부모 클래스의 포인터로 자식 클래스의 객체를 삭제할 때, 부모 클래스의 소멸자만 호출되면 자식 클래스의 자원은 해제되지 않음.  
- 예시:  

```cpp
class Base {
public:
    ~Base() {  // 가상 소멸자
        cout << "Base destructor called" << endl;
    }
};
class Derived : public Base {
public:
    ~Derived() {  // 소멸자
        cout << "Derived destructor called" << endl;
    }
};
int main() {
    Base* b = new Derived();  // Base 클래스 포인터로 Derived 객체 생성
    delete b;  // 올바른 소멸자 호출
    return 0;
}
```
- 위 예시에서, `Base` 클래스의 descructor가 virtual이 아니기 때문에, `delete b;`를 호출할 때 `Base` 클래스의 소멸자만 호출되고, `Derived` 클래스의 소멸자는 호출되지 않음.
- 따라서, `Base` 클래스의 소멸자는 virtual로 선언되어야 함.  

```cpp
class Base {
public:
    virtual ~Base() {  // 가상 소멸자
        cout << "Base destructor called" << endl;
    }
};
class Derived : public Base {
public:
    ~Derived() {  // 소멸자
        cout << "Derived destructor called" << endl;
    }
};
int main() {
    Base* b = new Derived();  // Base 클래스 포인터로 Derived 객체 생성
    delete b;  // 올바른 소멸자 호출
    return 0;
}
```
- 위 예시에서, `Base` 클래스의 소멸자가 virtual로 선언되어 있기 때문에, `delete b;`를 호출할 때 `Derived` 클래스의 소멸자가 먼저 호출되고, 그 다음에 `Base` 클래스의 소멸자가 호출됨. 이는 자식 클래스의 자원을 먼저 해제하고, 부모 클래스의 자원을 해제하기 위함임.

---
해당 포스트는 서울대학교 전기정보공학부 정교민 교수님의 프로그래밍방법론 25-1학기 강의를 정리한 내용입니다.
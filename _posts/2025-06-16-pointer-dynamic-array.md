---
layout: post
title: 'Pointer & Dynamic Array'
date: 2025-06-16 18:40 +0900
categories: ['C++']
tags: ['C++', '강의정리']
published: true
sitemap: true
math: true
---
## Pointer
C++에서 **포인터(Pointer)** 는 메모리 주소를 저장하는 변수로, 다른 변수의 메모리 주소를 가리킬 수 있음. 변수의 간접 참조에 사용됨.  

- 선언: `int* ptr;`
  - 주의: 
    - `int* ptr;`와 `int *ptr;`는 동일.
    - `int* p1, p2;`는 `p1`은 포인터, `p2`는 정수형 변수로 선언됨. 포인터 변수는 반드시 `*`와 함께 선언해야 함.

- 주소 연산자(`&`): 변수의 메모리 주소를 얻는 데 사용됨.
  - 예시: `int x = 10; int* ptr = &x;` // `ptr`은 `x`의 주소를 가리킴.

- 역참조 연산자(`*`): 포인터가 가리키는 주소의 값을 얻거나 변경하는 데 사용됨.
  - 예시: `int value = *ptr;` // `ptr`이 가리키는 주소의 값을 `value`에 저장.

- 포인터 대입
  - 포인터 간 대입: `ptr1 = ptr2;` // `ptr1`이 `ptr2`가 가리키는 주소를 가리킴.
  - 포인터 역참조 대입: `*ptr1 = *ptr2;` // `ptr1`이 가리키는 주소에 `ptr2`가 가리키는 값을 저장. 이 경우엔 한 포인터가 가리키는 값이 변경되어도 다른 포인터가 가리키는 값은 변경되지 않음.  

- `new` 연산자: 동적 메모리 할당에 사용됨.  
  - 동적 메모리 할당은 프로그램 실행 중에 메모리를 할당하고 해제할 수 있는 기능으로, 배열이나 객체를 동적으로 생성할 때 사용됨.
  - 프로그래머가 직접 메모리 사용을 관리할 수 있어 유연한 메모리 관리를 가능하게 함.
  - `new` 연산자를 만나면 운영체제는 heap 영역에서 필요한 크기의 메모리를 할당하고, 해당 메모리의 시작 주소를 반환함.
  - 동적으로 메모리를 관리하므로, 코드 블록이 끝나더라도 메모리가 유지됨.
  - 따라서 메모리 누수를 방지하기 위해 사용이 끝난 후에는 반드시 `delete` 연산자를 사용하여 메모리를 해제해야 함.
  - `delete` 으로 반납 후에도 포인터는 여전히 해당 메모리 주소를 가리키고 있으므로(dangling pointer), 이후에 해당 포인터를 사용하려면 `nullptr`로 초기화하는 것이 좋음.
  - heap에 공간이 없어 동적 할당이 실패할 경우, 옛날 컴파일러에서는 `nullptr`를 반환했지만, 최신 C++에서는 예외를 발생시킴.
  - 예시: 
    ```cpp
    int* ptr = new int; // 정수형 변수 동적 할당
    *ptr = 20; // 동적 할당된 변수에 값 대입
    delete ptr; // 동적 할당된 메모리 해제
    ptr = nullptr; // 포인터를 nullptr로 초기화
    ```

- Dynamic vs Automatic Variables

| 구분        | Dynamic Variables                          | Automatic Variables               |
| ----------- | ------------------------------------------ | --------------------------------- |
| 메모리 할당 | `new` 연산자를 사용하여 동적으로 할당      | 함수 호출 시 스택에 자동으로 할당 |
| 메모리 해제 | `delete` 연산자를 사용하여 명시적으로 해제 | 함수 종료 시 자동으로 해제        |

- Pointer Type definition: 포인터 타입을 정의할 때는 `typedef`를 사용하여 가독성을 높일 수 있음.  
  - 예시: 
    ```cpp
    typedef int* IntPtr; // IntPtr은 int형 포인터를 의미
    IntPtr p = new int; // IntPtr을 사용하여 동적 메모리 할당
    *p = 30; // 값 대입
    delete p; // 메모리 해제
    ```
- funtion with pointer argument: 포인터를 인자로 받는 함수는 포인터가 가리키는 값을 변경할 수 있음.  
  - 예시: 
    ```cpp
    void increment(int* ptr) {
        (*ptr)++; // 포인터가 가리키는 값 증가
    }
    
    int main() {
        int x = 5;
        increment(&x); // x의 주소를 전달
        std::cout << x; // 6 출력
        return 0;
    }
    ```
- function returning pointer: 함수가 포인터를 반환할 수 있음.  
  - 예시:  
    ```cpp
    int* createPtr() {
        int* ptr = new int; // 동적 메모리 할당
        *ptr = 10; // 값 대입
        return ptr; // 포인터 반환
    }
    ```
    이때, 정적 메모리 할당을 하면, 함수가 종료되면 해당 메모리가 해제되어 포인터가 가리키는 주소가 유효하지 않게 됨. 따라서 동적 메모리 할당을 사용해야 함.

- Pitfall: Call-by-Value  
    - Call-by-Value는 함수에 인자를 전달할 때, 인자의 값을 복사하여 전달하는 방식.
    - 이 방식은 함수 내부에서 인자의 값을 변경해도 원래 변수의 값은 변경되지 않음.
    - 하지만 포인터를 인자로 전달할 때, 함수 내부에서 포인터가 가리키는 값을 변경하면, 원래 포인터가 가리키는 값이 변경됨.

- Call-by-Reference as Call-by-Value with pointer
    - 포인터를 사용하여 Call-by-Reference처럼 동작하게 할 수 있음.
    - 함수에 포인터를 전달하면, 함수 내부에서 포인터가 가리키는 값을 변경할 수 있음.
    - 예시: 
      ```cpp
      void modifyValue(int* ptr) {
          *ptr = 20; // 포인터가 가리키는 값 변경
      }
      
      int main() {
          int x = 10;
          modifyValue(&x); // x의 주소를 전달
          std::cout << x; // 20 출력
          return 0;
      }
      ```
- 만약 한 클래스의 인스턴스를 다른 인스턴스로 복사할 때, 포인터 멤버가 있다면, 단순히 포인터 주소를 복사하는 것만으로는 원래 객체와 복사된 객체가 동일한 메모리를 가리키게 되어, 한 객체의 변경이 다른 객체에 영향을 미칠 수 있음. 이를 방지하기 위해 **깊은 복사(Deep Copy)**를 구현해야 함.  
  - 깊은 복사는 포인터가 가리키는 메모리까지 복사하여, 두 객체가 서로 독립적으로 동작하도록 함.
  - 예시: 클래스의 멤버 변수로 배열을 사용하는 경우
    ```cpp
    class MyClass {
    public:
        MyClass(int size) : size(size) {
            arr = new int[size]; // 동적 배열 할당
        }
        
        // 깊은 복사 생성자
        MyClass(const MyClass& other) : size(other.size) {
            arr = new int[size]; // 새로운 메모리 할당
            for (int i = 0; i < size; i++) {
                arr[i] = other.arr[i]; // 값 복사
            }
        }
        // 대입 연산자 오버로딩
        MyClass& operator=(const MyClass& other) {
            if (this != &other) { // 자기 자신과의 대입 방지
                delete[] arr; // 기존 메모리 해제
                size = other.size;
                arr = new int[size]; // 새로운 메모리 할당
                for (int i = 0; i < size; i++) {
                    arr[i] = other.arr[i]; // 값 복사
                }
            }
            return *this;
        }
        
        ~MyClass() {
            delete[] arr; // 메모리 해제
        }
    private:
        int* arr; // 동적 배열
        int size; // 배열 크기
    };
    ```

## Dynamic Array
- Array는 고정된 크기의 연속된 메모리 공간을 할당하여 데이터를 저장하는 자료구조로, C++에서는 배열의 크기를 컴파일 타임에 결정해야 함.
```cpp
int arr[5]; // 크기가 5인 정수형 배열 선언
```
- `arr` 자체는 배열의 시작 주소를 나타내는 포인터
- `arr = ptr;`와 같이 다른 포인터를 대입할 수 없음. 배열 이름은 상수 포인터로 취급되기 때문
- `arr`에 특정 값을 더하면 배열의 시작 주소에 해당 값만큼 오프셋을 더한 주소를 가리키게 됨.(곱셈, 나눗셈 연산은 불가능)  
  - 예시: `int* ptr = arr + 2;` // `arr`의 시작 주소에서 2번째 요소의 주소를 가리킴.  
  
```cpp
int arr[5] = {1, 2, 3, 4, 5};
int *ptr = arr;

int x = *(ptr + 2);  // 3
int y = *ptr;        // 1
ptr++;               // ptr은 arr[1] 가리킴
int z = *ptr;        // 2

cout << "x: " << x << ", y: " << y << ", z: " << z << endl; 
// 출력: x: 3, y: 1, z: 2
```

- **동적 배열(Dynamic Array)**: C++에서는 `new` 연산자를 사용하여 크기를 런타임에 결정할 수 있는 동적 배열을 생성할 수 있음.
  - 예시: 
    ```cpp
    int size;
    std::cout << "Enter size of array: ";
    std::cin >> size; // 사용자로부터 배열 크기 입력 받기
    int* arr = new int[size]; // 동적 배열 생성
    for (int i = 0; i < size; i++) {
        arr[i] = i + 1; // 배열 초기화
    }
    // 동적 배열 사용
    delete[] arr; // 동적 배열 메모리 해제
    arr = NULL; // 포인터를 nullptr로 초기화
    ```
  - heap에서 size만큼의 메모리를 할당하고, 해당 메모리의 시작 주소를 `arr` 포인터에 저장함.
  - 메모리 해제 시 `delete`가 아닌 `delete[]`를 사용해야 함. 이는 동적 배열이기 때문.

- Multi-dimensional Dynamic Array: C++에서는 다차원 배열을 동적으로 생성할 수 있음.  
  - 예시: 
    ```cpp
    int rows, cols;
    int** arr = new int*[rows]; // 행 포인터 배열 생성
    for (int i = 0; i < rows; i++) {
        arr[i] = new int[cols]; // 각 행에 대한 열 배열 생성
    }
    // 다차원 배열 사용
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[i][j] = i + j; // 배열 초기화
        }
    }
    // 메모리 해제
    for (int i = 0; i < rows; i++) {
        delete[] arr[i]; // 각 행의 열 배열 해제
    }
    delete[] arr; // 행 포인터 배열 해제
    arr = nullptr; // 포인터를 nullptr로 초기화
    ```

- array as return value: C++에서는 배열을 직접 반환할 수 없지만, 포인터를 사용하여 동적 배열을 반환할 수 있음.  
  - 예시: 
    ```cpp
    int* createArray(int size) {
        int* arr = new int[size]; // 동적 배열 생성
        for (int i = 0; i < size; i++) {
            arr[i] = i + 1; // 배열 초기화
        }
        return arr; // 동적 배열 반환
    }

    int main() {
        int size;
        cout << "Enter size of array: ";
        cin >> size; // 사용자로부터 배열 크기 입력 받기
        int* arr = createArray(size); // 동적 배열 생성 및 반환 받기
        // 동적 배열 사용
        cout << *(arr + 2) << endl; // 배열의 3번째 요소 출력
        cout << arr[0] << endl; // 배열의 첫 번째 요소 출력
        delete[] arr; // 동적 배열 메모리 해제
        arr = nullptr; // 포인터를 nullptr로 초기화
        return 0;
    }
    ```

---
해당 포스트는 서울대학교 전기정보공학부 정교민 교수님의 프로그래밍방법론 25-1학기 강의를 정리한 내용입니다.
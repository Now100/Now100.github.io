---
layout: post
title: 'Array & String'
date: 2025-04-17 13:43 +0900
categories: ['C++']
tags: ['C++', '강의정리']
published: true
sitemap: true
math: true
---
## Array
배열(Array)란 C++에서 여러 개의 동일한 타입의 데이터를 하나의 변수로 묶어서 관리할 수 있는 자료구조입니다. 배열은 고정된 크기를 가지며, 인덱스를 사용하여 각 요소에 접근할 수 있습니다.

```cpp
#include <iostream>
using namespace std;

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) {
        cout << arr[i] << " ";
    }
    return 0;
}
```

위 코드에서 `arr`은 정수형 배열이며, 5개의 요소를 포함하고 있습니다. 각 요소는 인덱스를 통해 접근할 수 있습니다. 인덱스는 항상 0부터 시작하고, 따라서 마지막 인덱스는 `arr의 크기 - 1`이 됩니다.

### Declaration
배열을 선언하는 방법은 다음과 같습니다:

```cpp
type ARRAY_NAME[ARRAY_SIZE];
```

예를 들어, 정수형 배열을 선언하려면 다음과 같이 작성할 수 있습니다:

```cpp
int numbers[10];
```

위 코드는 10개의 정수형 데이터를 저장할 수 있는 배열 `numbers`를 선언합니다. 배열의 크기는 반드시 상수여야 하며, 변수일 수 없습니다.

```cpp
int arrSize = 10;
int numbers[arrSize]; // compile error
```
```cpp
const int ARR_SIZE = 10;
int numbers[ARR_SIZE]; // valid use case
```
이때 선언된 배열의 원소 값은 해당 배열이 전역변수라면 0, 아니라면 쓰레기값으로 채워집니다. 

배열은 일반적인 변수와 같이 선언과 함께 초기화될 수 있습니다.

```cpp
int numbers[5] = {10, 20, 30, 40, 50};
```

배열의 요소는 부분적으로 초기화할 수도 있습니다. 초기화된 배열 요소 이외의 요소들은 기본값(0)으로 채워집니다.  

```cpp
int numbers[5] = {10, 20}; // == {10, 20, 0, 0, 0}
```

배열의 크기를 생략하고 초기화 리스트를 제공하면, 컴파일러가 배열의 크기를 자동으로 결정합니다.  

```cpp
int numbers[] = {1, 2, 3, 4, 5}; // Size is automatically set to 5
```

### Accessing Elements
배열의 요소에 접근하려면 배열 이름과 인덱스를 사용합니다. 예를 들어:

```cpp
int numbers[5] = {10, 20, 30, 40, 50};
cout << numbers[0]; // Outputs 10
cout << numbers[4]; // Outputs 50
```

C++의 컴파일러는 유효한 배열 인덱스를 벗어나는 범위 밖의 인덱싱을 컴파일 에러로 막지 않습니다.  

```cpp
#include <iostream>
using namespace std;

int main() {
    int numbers[3] = {1, 2, 3};
    cout << numbers[5]; // No compile error!
    return 0;
}
```

위 코드에서 `numbers[5]`는 배열의 범위를 벗어난 접근이지만, 컴파일 단계에서 오류로 걸러내지 않습니다. 따라서 의도치 않은 값을 반환하거나 예상치 못한 오류를 발생시킬 수 있으므로 사용에 주의해야합니다.  

### Memory structure of Array  

| **메모리 주소** | **배열 요소** | **값** |
| --------------- | ------------- | ------ |
| `0x7ffee3b0`    | `numbers[0]`  | `10`   |
| `0x7ffee3b4`    | `numbers[1]`  | `20`   |
| `0x7ffee3b8`    | `numbers[2]`  | `30`   |
| `0x7ffee3bc`    | `numbers[3]`  | `40`   |
| `0x7ffee3c0`    | `numbers[4]`  | `50`   |

배열은 메모리 상에서 연속적으로 저장되며, 각 요소는 데이터 타입 크기(int의 경우 보통 4바이트)를 기준으로 순차적으로 배치됩니다.

위 표의 예시에서 `numbers[i]`는 내부적으로 `*(numbers + i)`로 해석되어, 단순한 주소 연산을 통해 접근됩니다.

C++에서는 배열 범위를 벗어난 접근(`numbers[-1]`, `numbers[5]` 등)에 대해 컴파일러가 오류를 발생시키지 않으며, 해당 주소(`0x7ffee3ac`, `0x7ffee3c4`)에 실제로 어떤 값이 존재하는지는 보장되지 않습니다.

이는 **정의되지 않은 동작(undefined behavior)** 을 유발할 수 있으며, 프로그램의 오작동이나 예기치 않은 버그를 초래할 수 있습니다.

따라서 배열 접근 시에는 항상 인덱스의 범위를 명확히 확인하는 습관이 중요합니다.

### Function and Array
배열은 함수의 매개변수로 전달될 수 있습니다. 배열을 함수에 전달할 때, 배열의 크기 정보는 전달되지 않으므로, 함수 내부에서 배열의 크기를 알 수 없습니다. 따라서 배열과 함께 크기를 명시적으로 전달하는 것이 일반적입니다.

#### Passing Array to Function
배열을 Parameter로 선언할때는 다음과 같이 선언합니다:

```cpp
void functionName(type arrayName[], int size);
```

배열을 함수의 매개변수로 전달할 때, 배열의 첫 번째 요소의 주소가 전달되고, 이를 `arrayName[]`으로 표기합니다. 따라 배열의 크기 정보는 함수 내부에서 알 수 없으므로, 배열과 함께 크기를 명시적으로 전달해야 합니다.

예를 들어:

```cpp
#include <iostream>
using namespace std;

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main() {
    int numbers[5] = {10, 20, 30, 40, 50};
    printArray(numbers, 5);
    return 0;
}
```

위 코드에서 `printArray` 함수는 배열과 배열의 크기를 매개변수로 받아 배열의 요소를 출력합니다. 배열의 크기를 명시적으로 전달하지 않으면, 함수 내부에서 배열의 크기를 알 수 없으므로 의도치 않은 동작이 발생할 수 있습니다.

#### Array as Return Value

배열은 함수의 반환값으로 직접 사용할 수 없습니다. 대신, 배열을 반환하려면 포인터를 사용하거나 `std::array` 또는 `std::vector`와 같은 컨테이너를 사용하는 것이 일반적입니다.

### Multidimensional Array
다차원 배열(Multidimensional Array)은 배열의 배열로, 2차원 이상의 배열을 나타냅니다. 가장 일반적인 형태는 2차원 배열로, 행(row)과 열(column)로 구성됩니다.

#### Declaration
다차원 배열을 선언하는 방법은 다음과 같습니다:

```cpp
type ARRAY_NAME[ROW_SIZE][COLUMN_SIZE];
```

예를 들어, 3x4 크기의 정수형 2차원 배열을 선언하려면 다음과 같이 작성할 수 있습니다:

```cpp
int matrix[3][4];
```

#### Initialization
다차원 배열은 선언과 동시에 초기화할 수 있습니다. 초기화는 중괄호를 사용하여 각 행의 요소를 그룹화합니다.

```cpp
int matrix[2][3] = {
    {1, 2, 3},
    {4, 5, 6}
};
```

초기화시 차원생략은 오직 첫번째 차원에 대해서만 허용됩니다. 컴파일러는 `matrix[0][1]` 같은 요소에 접근할 때, 몇 개씩 건너뛰어야 할지를 계산해야 하기 때문입니다. 따라서 다차원 배열을 초기화할 시엔 항상 첫번째 차원을 제외하고는 차원을 명시해야합니다.

```cpp
int matrix[][3] = { // valid case
    {1, 2, 3},
    {4, 5, 6}
};

int tensor[][][] = { // invalid case
    {
        {1, 2, 3},
        {4, 5, 6}
    },
    {
        {7, 8, 9},
        {10, 11, 12}
    }
};
```

#### Accessing Elements
다차원 배열의 요소에 접근하려면 배열 이름과 각 차원의 인덱스를 사용합니다.

```cpp
int matrix[2][3] = {
    {1, 2, 3},
    {4, 5, 6}
};

cout << matrix[0][0]; // Outputs 1
cout << matrix[1][2]; // Outputs 6
```

다차원 배열의 모든 요소를 순회하려면 중첩된 반복문을 사용할 수 있습니다.

```cpp
#include <iostream>
using namespace std;

int main() {
    int matrix[2][3] = {
        {1, 2, 3},
        {4, 5, 6}
    };

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
```

위 코드에서 `i`는 행(row)을, `j`는 열(column)을 나타냅니다. 출력 결과는 다음과 같습니다:

```
1 2 3
4 5 6
```

#### Memory Structure of Multidimensional Array
다차원 배열은 메모리 상에서 연속적으로 저장됩니다. 예를 들어, `matrix[2][3]` 배열은 다음과 같이 메모리에 저장됩니다:

| 메모리 주소   | 배열 요소      | 값   |
| ------------- | ------------- | ---- |
| `0x7ffee3b0`  | `matrix[0][0]` | `1`  |
| `0x7ffee3b4`  | `matrix[0][1]` | `2`  |
| `0x7ffee3b8`  | `matrix[0][2]` | `3`  |
| `0x7ffee3bc`  | `matrix[1][0]` | `4`  |
| `0x7ffee3c0`  | `matrix[1][1]` | `5`  |
| `0x7ffee3c4`  | `matrix[1][2]` | `6`  |

배열의 요소는 행(row) 우선 순서로 저장됩니다. 즉, 첫 번째 행의 모든 요소가 저장된 후 두 번째 행의 요소가 저장됩니다.

#### Passing Multidimensional Array to Function
다차원 배열을 함수에 전달할 때, 초기화할때와 마찬가지로 두번째 차원부터 배열의 크기를 명시해야 합니다.

```cpp
void printMatrix(int matrix[][3], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < 3; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int matrix[2][3] = {
        {1, 2, 3},
        {4, 5, 6}
    };

    printMatrix(matrix, 2);
    return 0;
}
```

위 코드에서 `printMatrix` 함수는 2차원 배열과 행의 개수를 매개변수로 받아 배열의 요소를 출력합니다.

## Strings

C++에서 문자열(String)은 문자(char)의 배열로 표현됩니다. 문자열은 C 스타일 문자열(C-style string)과 C++ 표준 라이브러리의 `std::string` 클래스를 사용하여 처리할 수 있습니다.

### C-Strings

C-String은 `char` 데이터 타입의 배열입니다. C-String은 초기화시 자동으로 마지막에 null char `\0`를 포함합니다

#### Declaration and Initialization

```cpp
char str1[] = "Hello";
char str2[6] = "World"; // null 문자 포함 {'W', 'o', 'r', 'l', 'd', '\0'}
char str3[] = {'H', 'i', '\0'}; // 명시적 초기화
```

#### Accessing Characters

문자열의 각 문자는 배열처럼 인덱스를 사용하여 접근할 수 있습니다.

```cpp
#include <iostream>
using namespace std;

int main() {
    char str[] = "Hello";
    cout << str[0]; // Outputs 'H'
    cout << str[4]; // Outputs 'o'
    return 0;
}
```
#### Constraints on Operations
C-Strings는 배열이기 때문에, 배열과 관련된 제약 사항이 적용됩니다. 예를 들어, C-String을 다른 C-String에 직접 대입할 수 없습니다.

```cpp
char str1[] = "Hello";
char str2[] = "World";

str1 = str2; // Compile error
```

또한 C-String간 서로 값을 비교할 때에도, 비교 연산자는 두 문자열의 포인터간 비교를 할 뿐, 그 값에 대해 비교를 해주지 않습니다. 따라서 같은 내용의 두 C-String이더라도 `==` 연산시 `true`를 반환하지 않습니다.

```cpp
char str1[] = "Hello";
char str2[] = "Hello";

str1 == str2; // false
```

이와 같은 이유로 인해, C-string에 대해 연산을 하기 위해선 사전 정의된 함수를 이용해야합니다. 

#### Functions for C-Strings
```cpp
#include <cstring>
#include <iostream>
using namespace std;

int main() {
    char str1[] = "Hello";
    char str2[] = "World";

    // Copying strings
    strcpy(str1, str2); // str1 now contains "World"
    cout << "After strcpy: " << str1 << endl;

    // Concatenating strings
    strcat(str1, "!!!"); // str1 now contains "World!!!"
    cout << "After strcat: " << str1 << endl;

    // Comparing strings
    int result = strcmp(str1, str2); // Compares str1 and str2
    if (result == 0) {
        cout << "Strings are equal" << endl;
    } else if (result < 0) {
        cout << "str1 is less than str2" << endl;
    } else {
        cout << "str1 is greater than str2" << endl;
    }

    // Finding string length
    cout << "Length of str1: " << strlen(str1) << endl;

    return 0;
}
```

위 코드에서 사용된 주요 함수는 다음과 같습니다:

- `strcpy(destination, source)`  
    `source` 문자열을 `destination` 문자열 에 복사합니다.   
    이때 `destination` 문자열이 `source`보다 짧다면 배열 크기를 넘어선 할당이 일어나므로 주의해야합니다.   
- `strcat(destination, source)`  
    `destination` 문자열의 끝(null 문자의 앞)에 `source` 문자열을 추가합니다. 
    이 함수 역시 `destination` 문자열이 짧다면 배열 크기를 넘어선 할당이 일어나므로 주의해야합니다.   
- `strcmp(str1, str2)`  
    두 문자열을 비교합니다. 반환값이 0이면 두 문자열이 같고, 음수이면 `str1`이 `str2`보다 작고(작다는 것은 ASCII code 상 앞에 나온다는 뜻) 양수이면 `str1`이 `str2`보다 큽니다.
- `strlen(str)`  
    문자열의 길이를 반환합니다(널 문자는 포함하지 않음).

C-String 배열의 특정 부분만 연산에 활용하는 함수도 있습니다.  
- `strncpy(destination, source, n)`  
    `source` 문자열의 처음 `n`개의 문자를 `destination`의 첫 `n`개 문자에 복사합니다.  
    이 함수 역시 `destination`이 `n`보다 작으면 배열의 크기를 넘어서 저장되기 때문에 주의해야합니다. 

- `strncat(destination, source, n)`  
    `destination` 문자열의 끝에 `source` 문자열의 처음 `n`개의 문자를 추가합니다. 이 함수 역시 최종 결과값이 `destination`의 크기를 초과하지 않도록 주의해야 합니다.  

- `strncmp(str1, str2, n)`  
    두 문자열의 처음 `n`개의 문자를 비교합니다. 반환값은 `strcmp`와 동일합니다.

```cpp
#include <cstring>
#include <iostream>
using namespace std;

int main() {
        char str1[20] = "Hello";
        char str2[] = "World";

        // Copying first 3 characters
        strncpy(str1, str2, 3); // str1 now contains "Worlo"
        cout << "After strncpy: " << str1 << endl;

        // Concatenating first 2 characters
        strncat(str1, "ld!", 2); // str1 now contains "Worlold"
        cout << "After strncat: " << str1 << endl;

        // Comparing first 3 characters
        int result = strncmp(str1, str2, 3);
        if (result == 0) {
                cout << "First 3 characters are equal" << endl;
        } else if (result < 0) {
                cout << "First 3 characters of str1 are less than str2" << endl;
        } else {
                cout << "First 3 characters of str1 are greater than str2" << endl;
        }
        // First 3 characters are equal

        return 0;
}
```

- **cout and null char(`\0`)**  
    C++에서 `cout`은 문자열을 출력할 때 null 문자(`\0`)를 만나면 출력을 중단합니다. 이는 null 문자가 문자열의 끝을 나타내는 표준 C 스타일 문자열의 특성 때문입니다.

    ```cpp
    #include <iostream>
    using namespace std;

    int main() {
        char str[] = "Hello\0World";
        cout << str << endl; // Outputs "Hello"
        return 0;
    }
    ```

    위 코드에서 `str`은 "Hello\0World"로 초기화되었지만, `cout`은 null 문자(`\0`)를 만나면 출력을 중단하므로 "Hello"만 출력됩니다.  

    따라서 어떤 문자열에 null 문자가 포함되지 않는다면 출력이 끝나지 않고 의미없는 garbage들이 출력되는 현상이 나타날 수 있습니다.  

    null 문자는 문자열의 끝을 나타내기 때문에, 문자열의 길이를 계산하거나 문자열을 조작할 때 중요한 역할을 합니다. 예를 들어, `strlen` 함수는 null 문자를 만나면 문자열의 길이 계산을 중단합니다.  

    ```cpp
    #include <iostream>
    #include <cstring>
    using namespace std;

    int main() {
        char str[] = "Hello\0World";
        cout << "Length of str: " << strlen(str) << endl; // Outputs 5
        return 0;
    }
    ```

    위 코드에서 `strlen(str)`은 null 문자 이전의 문자만 계산하므로 문자열의 길이를 5로 반환합니다.  

### `std::string`
`std::string`은 C++ 표준 라이브러리에서 제공하는 문자열 클래스입니다. 이는 C-String보다 사용하기 쉽고, 다양한 기능을 제공합니다.

#### Declaration and Initialization

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    string str1 = "Hello";
    string str2("World");
    string str3;
    cout << str1 << endl; // Outputs "Hello"
    cout << str2 << endl; // Outputs "World"
    cout << str3 << endl; // Outputs ""
    return 0;
}
```
`int`, `double`과 같이 쉽게 선언과 초기화가 가능합니다. 위 예시의 `str1`과 같이 assignment를 이용해 초기화할 수도 있고, `str2`와 같이 constructor를 이용해 초기화할 수도 있습니다.    
만약 `str3`처럼 선언만 해도, 빈 문자열("")로 자동 초기화되고, 쓰레기값이나 미정의 상태로 정해지지 않습니다.

#### Accessing Characters

`string`은 배열처럼 인덱스를 사용하여 각 문자에 접근할 수 있습니다.

```cpp
string str = "Hello";
cout << str[0]; // Outputs 'H'
cout << str.at(4); // Outputs 'o'
```

#### Common Operations

- **Length**(`.length()`): 문자열의 길이를 반환합니다.
  ```cpp
  string str = "Hello";
  cout << str.length(); // Outputs 5
  ```

- **Concatenation**: 문자열을 `+` 연산자를 활용해 결합합니다.
  ```cpp
  string str1 = "Hello";
  string str2 = "World";
  string str3 = str1 + " " + str2;
  cout << str3; // Outputs "Hello World"
  ```

- **Comparison**: 기본 비교 연산자를 이용해 문자열을 비교합니다.
  ```cpp
  string str1 = "Hello";
  string str2 = "World";
  if (str1 == str2) {
      cout << "Strings are equal";
  } else {
      cout << "Strings are not equal";
  }
  ```
  여기서 '크고 작다'는 마찬가지로 사전순 앞/뒤를 의미합니다(ASCII code).

- **Substring**(`.substr()`): 문자열의 일부를 추출합니다.
  ```cpp
  string str = "Hello World";
  string sub = str.substr(0, 5); // Extracts "Hello"
  cout << sub;
  ```

- **Find**(`.find()`): 특정 문자열이나 문자의 위치를 찾습니다.
  ```cpp
  string str = "Hello World";
  size_t pos = str.find("World");
  if (pos != string::npos) {
      cout << "Found at position: " << pos;
  }
  ```

- **Replace**(`.replace()`): 문자열의 일부를 다른 문자열로 대체합니다.
  ```cpp
  string str = "Hello World";
  str.replace(6, 5, "C++");
  cout << str; // Outputs "Hello C++"
  ```

#### Null in `string`
`std::string` 클래스는 null 문자(`\0`)를 포함할 수 있습니다. 그러나 `std::string`은 null 문자를 특별히 처리하지 않으며, 문자열의 길이를 계산하거나 출력할 때 null 문자를 포함한 전체 문자열을 처리합니다.

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    string str = "Hello\0World";
    cout << "String: " << str << endl; // Outputs "Hello"
    cout << "Length: " << str.length() << endl; // Outputs 11
    return 0;
}
```

위 코드에서 `str`은 null 문자를 포함하지만, `std::string`은 null 문자를 문자열의 끝으로 간주하지 않습니다. 따라서 `str.length()`는 전체 문자열의 길이(11)를 반환합니다.

이와 같은 특성은 C-String과의 주요 차이점 중 하나입니다. C-String은 null 문자를 문자열의 끝으로 간주하지만, `std::string`은 null 문자를 일반 문자로 처리합니다.

### `getline`

C++에서 문자열 입력을 받을 때 `cin`은 기본적으로 공백을 기준으로 입력을 나눠 받습니다. 하지만 line 단위로 입력을 받아야할 상황도 있는데요, 이를 위해 C-String과 `std::string`은 각각 `cin.getline`과 `std::getline`을 사용합니다. 두 함수는 비슷한 역할을 하지만, 사용법과 동작 방식에서 차이가 있습니다.


#### `cin.getline` (C-String)

`cin.getline`은 C-String에 입력을 저장하는 함수입니다. 사용법은 다음과 같습니다:

```cpp
cin.getline(char* str, int size, char delimiter = '\n');
```

- `str`: 입력을 저장할 C-String 배열의 포인터.
- `size`: 입력받을 최대 문자 수 (널 문자 포함).
- `delimiter`: 입력 종료를 나타내는 구분 문자 (기본값은 `\n`).

```cpp
#include <iostream>
using namespace std;

int main() {
    char str[50];
    cout << "Enter a string: ";
    cin.getline(str, 50);
    cout << "You entered: " << str << endl;
    return 0;
}
```

- 입력이 `size - 1` 문자를 초과하면 나머지 입력은 버퍼에 남습니다.
- 구분 문자(`delimiter`)를 만나면 입력을 종료하고, 구분 문자는 버퍼에서 제거됩니다.
- 입력받는 배열의 크기를 초과하지 않도록 주의해야 합니다.

#### `std::getline` (`std::string`)

`std::getline`은 `std::string` 객체에 입력을 저장하는 함수입니다. 사용법은 다음과 같습니다:

```cpp
std::getline(std::istream& is, std::string& str, char delimiter = '\n');
```

- `is`: 입력 스트림 (일반적으로 `std::cin`).
- `str`: 입력을 저장할 `std::string` 객체.
- `delimiter`: 입력 종료를 나타내는 구분 문자 (기본값은 `\n`).

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    string str;
    cout << "Enter a string: ";
    getline(cin, str);
    cout << "You entered: " << str << endl;
    return 0;
}
```

- 입력 크기에 제한이 없으며, 문자열 크기가 자동으로 조정됩니다.
- 구분 문자(`delimiter`)를 만나면 입력을 종료하고, 구분 문자는 버퍼에서 제거됩니다.
- 메모리 관리가 자동으로 이루어지므로 더 안전하고 편리합니다.

---

#### 비교

| Feature                  | `cin.getline` (C-String)                  | `std::getline` (`std::string`)       |
|--------------------------|-------------------------------------------|--------------------------------------|
| **Input Storage**        | `char` 배열                              | `std::string` 객체                  |
| **Memory Management**    | 수동 (배열 크기 지정 필요)               | 자동 (크기 제한 없음)               |
| **Delimiter Handling**   | 구분 문자 제거                           | 구분 문자 제거                      |
| **Buffer Overflow**      | 배열 크기 초과 시 위험                   | 없음 (자동 크기 조정)               |
| **Ease of Use**          | 복잡 (배열 크기와 포인터 관리 필요)      | 간단 (자동 메모리 관리)             |
| **Flexibility**          | 고정된 크기의 입력만 처리 가능           | 가변 크기의 입력 처리 가능          |

---

#### 버퍼 문제  
`cin`과 `getline`을 혼용할 경우, 입력 버퍼에 남아 있는 `\n` 문자로 인해 예상치 못한 동작이 발생할 수 있습니다. 이를 방지하려면 `cin.ignore()`를 사용하여 버퍼를 비워야 합니다.

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    int num;
    string str;

    cout << "Enter a number: ";
    cin >> num;
    cin.ignore(); // Clear the newline character from the buffer

    cout << "Enter a string: ";
    getline(cin, str);

    cout << "Number: " << num << ", String: " << str << endl;
    return 0;
}
```

위 코드는 `cin`과 `getline`을 혼용하여 입력을 처리하는 예제입니다. 이 경우, `cin`으로 숫자를 입력받은 후 입력 버퍼에 남아 있는 개행 문자(`\n`)가 `getline` 함수에 의해 처리될 수 있습니다.   
`cin`은 보통 버퍼에 남아있는 `\n`을 무시하고 입력을 받는 반면, `getline`이 개행 문자를 입력 종료로 간주하기 때문입니다.

1. `cin`은 숫자 입력 후 개행 문자(`\n`)를 버퍼에 남깁니다.
2. `getline`은 입력을 받을 때 버퍼에 남아 있는 개행 문자를 만나 즉시 종료됩니다.
3. 결과적으로, 사용자가 문자열을 입력하기 전에 `getline`이 호출되어 빈 문자열을 반환합니다.

따라서 `cin.ignore()`를 사용하여 입력 버퍼에 남아 있는 개행 문자를 제거하면 문제를 해결할 수 있습니다. `cin.ignore()`는 지정된 개수의 문자를 무시하거나, 기본적으로 첫 번째 개행 문자까지 무시합니다.

### How to use 한글 in C++
C++에서 한글을 문자열로 사용할 때, 유니코드(UTF-16 또는 UTF-32)를 사용해야합니다. 이를 위해 `wchar_t` 타입과 `wchar.h` 헤더를 사용할 수 있습니다. `wchar_t`는 넓은 문자(wide character)를 나타내며, 유니코드 문자를 저장할 수 있습니다.

```cpp
#include <iostream>
#include <wchar.h>
using namespace std;

int main() {
    wchar_t str[] = L"안녕하세요"; // L prefix indicates a wide string
    wcout << L"Wide string: " << str << endl;
    return 0;
}
```

1. `L prefix`를 통해 wide string 리터럴을 나타냅니다.
2. `cout`이 아닌 `wcout`을 사용해야만 `wchar_t`를 출력할 수 있습니다.

---
해당 포스트는 서울대학교 전기정보공학부 정교민 교수님의 프로그래밍방법론 25-1학기 강의를 정리한 내용입니다.
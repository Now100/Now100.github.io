---
layout: post
title: 'Git & GitHub 사용법'
date: 2025-04-17 17:11 +0900
categories: ['git']
tags: ['git', 'github', 'version control']
published: true
sitemap: true
math: true
---
## Git
- **Git이란?**  
    Git은 분산 버전 관리 시스템(DVCS)으로, 소스 코드의 변경 사항 등 전체 프로젝트 기록을 로컬 저장소에 저장하고 여러 개발자 간의 협업을 가능하게 합니다. 빠르고 효율적인 브랜치 관리와 병합 기능을 제공합니다.

- **Git의 유용성**
    - 버전 관리 (Version Control)
        - 파일 복사로 버전 관리 → 너무 복잡, 충돌 발생
        - Git으로는 효율적인 변경 추적가
    - 백업 (Back up)
        - 시스템 오류 시 과거 버전으로 복원 가능
    - 협업 (Collaboration)
        - 여러 개발자가 동시에 작업 가능
        - 변경사항 기록 및 추적 용이
        - 여러 저장소 동시 작업 가능

- `git init`
    - 해당 명령어로 새로운 Git 저장소를 초기화합니다.
    - 실행 후, `.git` 디렉토리가 생성되며 해당 디렉토리가 Git 저장소로 설정됩니다.


### 파일의 세가지 상태
1. **Untracked/Modified**  
    - Git이 관리하지 않거나, Git 저장소에 저장된 후의 수정사항이 반영되지 않은 파일 상태입니다.  
    - `git add` 명령어를 사용하여 Staging Area로 이동시킬 수 있습니다.

2. **Staged**  
    - 커밋 준비가 된 상태로, Staging Area에 있는 파일입니다.  
    - `git commit` 명령어를 사용하여 로컬 저장소에 저장할 수 있습니다.
    - 커밋 시에는 `-m`을 통해 반드시 커밋메세지를 작성해야합니다.

3. **Committed**  
    - 로컬 저장소에 저장된 상태입니다.  
    - 안전하게 버전 관리되고 있으며, 필요 시 과거 상태로 복원 가능합니다.

### Branching & Merging
- **Branch**  
    - 브랜치는 독립적인 작업 공간을 제공합니다.  
    - `git branch <branch_name>` 명령어로 새로운 브랜치를 생성할 수 있습니다.  

- **Merge**  
    - 다른 브랜치에서 작업한 결과를 현재 브랜치에 통합합니다.  
    - `git merge <branch_name>` 명령어를 사용하여 병합을 수행합니다.  

### Reset & Checkout
- **Reset**  
    - `git reset <commit>` 명령어는 현재 커밋을 특정 커밋으로 되돌리거나, Staging Area 또는 작업 디렉토리의 변경 사항을 초기화하는 데 사용됩니다.  
    - 주요 옵션:
        - `--soft`: 커밋만 되돌리고 Staging Area는 유지합니다.
        - `--mixed`: 커밋과 Staging Area를 되돌리고 작업 디렉토리는 유지합니다.
        - `--hard`: 커밋, Staging Area, 작업 디렉토리를 모두 되돌립니다.

    

- **Checkout**  
    - `git checkout` 명령어는 브랜치를 전환하거나 특정 커밋으로 작업 디렉토리를 업데이트하는 데 사용됩니다.  
    - 주요 사용법:
        - `git checkout <branch_name>`: 지정된 브랜치로 전환합니다.
        - `git checkout <commit_hash>`: 특정 커밋으로 작업 디렉토리를 업데이트합니다.  
        - `git checkout -b <new_branch_name>`: 새로운 브랜치를 생성하고 전환합니다.

### Conflict
- **충돌(Conflict)이란?**  
    - 서로 다른 브랜치에서 동일한 파일의 동일한 부분을 수정하고 두 브랜치를 merge하려고 하면 발생합니다.  
    - Git은 충돌이 발생한 파일을 자동으로 병합하지 못하며, 수동으로 해결해야 합니다.

- **충돌 해결 방법**  
    1. 충돌 파일 확인  
        - `git status` 명령어를 사용하여 충돌이 발생한 파일을 확인합니다.
    2. 충돌 내용 확인  
        - 충돌 파일을 열어 `<<<<<<<`, `=======`, `>>>>>>>`로 표시된 충돌 부분을 확인합니다.
    3. 충돌 해결  
        - 충돌 부분을 수정하여 원하는 내용을 반영합니다.
    4. 수정 사항 반영  
        - 수정된 파일을 Staging Area로 추가합니다: `git add <file_name>`.
    5. 병합 완료  
        - `git commit` 명령어를 사용하여 병합을 완료합니다.

#### 예시
1. **충돌 상황 만들기**  
    - 두 명의 개발자가 동일한 파일을 수정한다고 가정합니다.  
    - 예를 들어, `example.txt` 파일의 내용을 아래와 같이 수정합니다.

    **개발자 A의 수정 내용:**
    ```text
    Hello, this is Developer A's change.
    ```

    **개발자 B의 수정 내용:**
    ```text
    Hello, this is Developer B's change.
    ```

2. **충돌 발생**  
    - 개발자 A가 먼저 변경 사항을 커밋합니다:
        ```bash
        git add example.txt
        git commit -m "Developer A's change"
        ```

    - 개발자 B가 변경 사항을 커밋하려고 하면 충돌이 발생합니다:
        ```bash
        git add example.txt
        git commit -m "Developer B's change"
        ```
        ```text
        error: Commit your changes or stash them before you merge.
        ```

3. **충돌 해결 과정**  

    - 충돌이 발생한 파일을 확인합니다:
        ```bash
        git status
        ```
        ```text
        both modified: example.txt
        ```

    - `example.txt` 파일을 열어 충돌 내용을 확인합니다:
        ```text
        <<<<<<< HEAD
        Hello, this is Developer B's change.
        =======
        Hello, this is Developer A's change.
        >>>>>>> main
        ```

    - 충돌 부분을 수정하여 최종 내용을 결정합니다:
        ```text
        Hello, this is the final agreed change.
        ```

4. **수정 사항 반영 및 병합 완료**  
    - 수정된 파일을 Staging Area로 추가합니다:
        ```bash
        git add example.txt
        ```

    - 병합을 완료합니다:
        ```bash
        git commit -m "Resolve merge conflict in example.txt"
        ```

## Github
- GitHub는 Git 저장소를 호스팅하는 클라우드 기반 플랫폼으로, 개발자들이 협업하고 코드를 공유할 수 있도록 지원합니다.  
- 주요 기능:
    - 원격 저장소 관리
    - 협업 도구 (Pull Request, Issues 등)
    - CI/CD 통합
    - 프로젝트 관리 (Projects, Wiki 등)

### GitHub의 주요 명령어
1. **`git clone`**
    - 원격 저장소를 로컬로 복제합니다.
    ```bash
    git clone <repository_url>
    ```

2. **`git remote`**
    - 원격 저장소를 관리합니다.
    - 원격 저장소 추가:
        ```bash
        git remote add origin <repository_url>
        ```
    - 원격 저장소 확인:
        ```bash
        git remote -v
        ```

3. **`git push`**
    - 로컬 변경 사항을 원격 저장소에 업로드합니다.
    ```bash
    git push origin <branch_name>
    ```

4. **`git pull`**
    - 원격 저장소의 변경 사항을 로컬로 가져옵니다.
    ```bash
    git pull origin <branch_name>
    ```

5. **`git fetch`**
    - 원격 저장소의 변경 사항을 가져오지만, 로컬 브랜치에는 병합하지 않습니다.
    ```bash
    git fetch origin
    ```

### GitHub Flow
1. **Main 브랜치**
    - 항상 배포 가능한 상태를 유지합니다.

2. **Feature 브랜치**
    - 새로운 기능 개발을 위해 생성합니다.
    ```bash
    git checkout -b feature/<feature_name>
    ```

3. **Pull Request**
    - Feature 브랜치를 Main 브랜치에 병합하기 전에 코드 리뷰를 요청합니다.
    - GitHub 웹 인터페이스에서 생성 가능합니다.

4. **Merge**
    - 코드 리뷰가 완료되면 Pull Request를 병합합니다.
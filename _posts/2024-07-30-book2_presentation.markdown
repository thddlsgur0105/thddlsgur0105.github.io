---
layout: post
title:  "7.30 발표자료"
date:   2024-07-28 21:14:31 +0900
categories: ML Session
toc: true
pin: true
---

# 01. 나의 첫 머신러닝

## 01-1 인공지능과 머신러닝, 딥러닝

* 인공지능: 사람처럼 학습하고 추론할 수 있는 지능을 가진 컴퓨터 시스템을 만드는 기술
    * **인공일반지능** 혹은 **강인공지능**: 흔히 영화 속의 인공지능
    * **약인공지능**: 현실에서 우리가 마주하고 있는 인공지능으로, 특정 분여에서 사람의 일을 도와주는 보조 역할만 가능하다.

---    

* 머신러닝: 규칙을 일일히 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘을 연구하는 분야이다.
    * 통계학에서 유래된 머신러닝 알고리즘이 많으며 통계학과 컴퓨터 과학 분야가 상호 작용하면서 발전하고 있다.
    * 사이킷런: 컴퓨터 과학 분야의 대표적인 머신러닝 라이브러리
    

---

* 딥러닝: 많은 머신러닝 알고리즘 중에 **인공 신경망**을 기반으로 한 방법들을 통칭하여 부르는 말
* 딥러닝 라이브러리: 구글의 *텐서플로*, 페이스북의 *파이토치*


## 01-2 코랩과 주피터 노트북
* 코랩: 구글 계정이 있으면 누구나 사용할 수 있는 **웹 브라우저 기반의 파이썬 코드 실행 환경**
* 셀: 코랩에서 실행할 수 있는 **최소 단위**
* 노트북: 코랩의 프로그램 작성 단위이며 일반 프로그램 파일과 달리 대화식으로 프로그램을 만들 수 있다. 코드, 코드의 실행 결과, 문서를 모두 저장하여 보관 가능
* 코랩 노트북은 **구글 클라우드의 가상 서버를 사용**함

## 01-3 마켓과 머신러닝

### 생선 분류 문제
> 생선의 길이가 30cm 이상이면 도미이다.
```python
# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```

* 분류: 머신러닝에서 여러 개의 종류(혹은 클래스) 중 하나를 구별하는 문제
    * 이진 분류: 2개의 클래스 중 하나를 고르는 문제

---

* 특성(feature): 데이터의 특징
```python
# 첫 번째 도미
bream_length[0] = 25.4 # 특성 1. 도미의 길이
bream_weight[0] = 242.0 # 특성 2. 도미의 무게
```
---

* 산점도: $x$, $y$축으로 이뤄진 좌표계에 두 변수의 관계를 표현하는 방법
    ```python
    import matplotlib.pyplot as plt

    plt.scatter(bream_length, bream_weight) # 도미 데이터 
    plt.scatter(smelt_length, smelt_weight) # 빙어 데이터
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    ```


---
### 첫 번째 머신러닝 프로그램

```python
length = bream_length + smelt_length
#  [ length[0], length[35] ) : 도미(bream) 35개 길이
#  [ length[35], length[49] ) : 빙어(smelt) 14개 길이
weight = bream_weight + smelt_weight
#  [ weight[0], weight[35] ) : 도미(bream) 35개 무게
#  [ weight[35], weight[49] ) : 빙어(smelt) 14개 무게
``` 

```python
fish_data = [[l, w] for (l, w) in zip(length, weight)]
# fish_data = [ [length[0], weight[0]], [length[1], weight[1]] ... ]

# 리스트 내포(list comprehension) 구문
# zip() 함수는 나열된 리스트에서 원소를 하나씩 꺼내주는 역할
```

```python
fish_target=[1]*35+[0]*14 # 타겟리스트(정답리스트) 생성
```
---
* 파이썬에서 패키지나 모듈 전체를 임포트하지 않고 <u>특정 클래스만 임포트</u>하려면 **from ~import** 구문을 사용한다.

```python
# k-최근접 이웃 알고리즘 클래스 import
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
```
---
* 모델에 데이터를 전달하여 규칙을 학습하는 **훈련**과정을 거친다.
```python
# 모델 kn에 fish_data, fish_target을 전달해 훈련
kn.fit(fish_data, fish_target)
```

```python
# 훈련이 끝난 모델에 테스트에 사용할 데이터를 넘겨주어 모델을 평가
kn.score(fish_data, fish_target) # 1.0
```

```python
# 훈련이 끝난 모델에 x값(length), y값(weight) 리스트를 주어 결과 예측(1 또는 0)
kn.predict([[30, 600], [29, 599], [10, 100]]) # array([1, 1, 0])
```
---

```python
# 가장 가까운 데이터 49개를 사용하는 k-최근접 이웃 모델
kn_49 = KNeighborsClassifier(n_neighbors=49)

kn_49.fit(fish_data, fish_target)

kn_49.score(fish_data, fish_target) # 0.7142857142857143 ~= 35/49
```

#### k-최근접 이웃 알고리즘이란?
* 어떤 데이터에 대한 답을 구할 때 주위의 다른 데이터를 보고 **다수를 차지하는 것을 정답**으로 사용한다.

* **데이터가 아주 많은 경우 사용하기 어렵다.** 
    * 데이터가 크기 때문에 메모리가 많이 필요하고 <u>직선거리를 계산하는 데도 많은 시간이 필요하기 때문이다.</u>
* 사실 어떤 규칙을 찾기보다는 **전체 데이터를 메모리에 가지고 있는 것이 전부이다.**

---
* 맨허튼 거리와 유클리드 거리
    * 빨강, 파랑, 노랑: 맨허튼 거리 / 초록: 유클리드 거리


<br></br>

# 02. 데이터 다루기

## 02-1. 훈련 세트와 테스트 세트

### 지도 학습과 비지도 학습
* 지도 학습 알고리즘은 <u>입력(데이터)과 타깃(정답)으로 이뤄진 훈련 데이터가 필요합니다.</u>
* 비지도 학습 알고리즘은 <u>타깃(정답) 없이 입력 데이터만 사용합니다.</u>

* 비지도 학습은 타깃 데이터가 없다. 따라서 무엇을 예측하는 것이 아니라 어떤 특징을 찾는 데 주로 활용한다.

## 훈련 세트와 테스트 세트
* 평가에 사용하는 데이터를 **테스트 세트**, 훈련에 사용되는 데이터를 **훈련 세트**라고 부릅니다.

* 훈련 세트는 모델을 훈련할 때 사용하는 데이터로, <u>보통 훈련 세트가 클수록 좋다.</u> 따라서 테스트 세트를 제외한 모든 데이터를 사용한다.

* 테스트 세트는 전체 데이터에서 20~30%를 테스트 세트로 사용하는 경우가 많다. 전체 데이터가 아주 크다면 1%만 덜어내도 충분할 수 있다.

```python
# 슬라이싱 연산으로 처음 35개 샘플은 훈련 세트로, 나머지 14개 샘플은 테스트 세트로 선택
train_input = fish_data[:35]
train_target = fish_target[:35]

test_input = fish_data[35:]
test_target = fish_target[35:]
```

```python
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target) # 0.0
```

## 샘플링 편향
* 일반적으로 <u>훈련 세트와 테스트 세트에 샘플이 골고루 섞여 있지 않으면</u> 샘플링이 한 쪽으로 치우쳤다는 의미로 **샘플링 편향**이라고 부른다.
* 특정 종류의 샘플이 과도하게 많은 샘플링 편향을 가지고 있다면 제대로 된 지도 학습 모델을 만들 수 없다.

    * 올바른 훈련 데이터
    

    * 잘못된 훈련 데이터
    

## 넘파이
* 파이썬의 대표적인 배열 라이브러리
* 고차원의 배열을 손쉽게 만들고 조작할 수 있는 간편한 도구를 많이 제공
* 배열 인덱싱: 1개의 인덱스가 아닌 여러 개의 인덱스로 한 번에 여러 개의 원소를 선택할 수 있다.

```python
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr)

# input_arr
# [[  25.4  242. ]
# [  26.3  290. ]
# [  26.5  340. ]
# ...
# [  15.    19.9]]

print(target_arr)

# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0]
```

```python
np.random.seed(42)
index = np.arange(49) # index = [0, 1, 2, 3, ..., 47, 48]
np.random.shuffle(index)

print(index)
# [13 45 47 44 17 27 26 25 31 19 12  4 34  8  3  6 40 41 46 15  9 16 24 33
#  30  0 43 32  5 29 11 36  1 21  2 37 35 23 39 10 22 18 48 20  7 42 14 28 38]
```

```python
# 배열 인덱싱으로 랜덤 셔플한 인덱스의 처음 35개는 훈련 세트로,
# 나머지 14개는 테스트 세트로 구성
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
```

```python
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


### 핵심 패키지와 함수
* $seed()$: 넘파이에서 난수를 생성하기 위한 정수 초깃값을 지정한다. 초깃값이 같으면 동일한 난수를 뽑을 수 있다. 따라서 랜덤 함수의 결과를 동일하게 재현하고 싶을 때 사용한다.

* $arange()$: 일정한 간격의 정수 또는 실수 배열을 만든다. 기본 간격은 1이다. 매개변수가 하나이면 종료 숫자를 의미한다. 0에서 종료 숫자까지의 배열을 만들고, 종료 숫자는 배열에 포함되지 않는다.

* $shuffle()$: 주어진 배열을 랜덤하게 섞는다. 다차원 배열의 경우 첫 번째 축(행)에 대해서 섞는다.

## 두 번째 머신러닝 프로그램

```python
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target) # 1.0

kn.predict(test_input)
# array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])

test_target
# array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
```
* $predict()$ 메서드가 반환하는 값은 단순한 파이썬 리스트가 아니라 <u>넘파이 배열</u>이다.


## 02-2. 데이터 전처리
> 좀 문제가 있는데, 길이가 25cm이고 무게가 250g이면 도미인데 자네 모델은 빙어라고 예측한다는군.

### 넘파이로 데이터 준비하기
```python
# 도미와 빙어에 대한 fish_length, fish_weight의 특성 데이터 준비
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```

```python
import numpy as np

fish_data = np.column_stack((fish_length, fish_weight))

print(fish_data[:5])
# [[ 25.4 242. ]
#  [ 26.3 290. ]
#  [ 26.5 340. ]
#  [ 29.  363. ]
#  [ 29.  430. ]]

fish_target = np.concatenate((np.ones(35), np.zeros(14)))

print(fish_target)
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```

* 튜플($tuple$): 리스트와 매우 비슷하지만 <u>리스트처럼 원소에 순서는 있지만 한 번 만들어진 튜플은 수정이 불가능</u>하다.
* 튜플을 사용하면 함수로 전달한 값이 바뀌지 않는다는 것을 믿을 수 있기 때문에 매개변수 값으로 많이 사용한다.
* <u>데이터가 클수록 파이썬 리스트는 비효율적이므로 넘파이 배열을 사용하는 게 좋다</u>.

### 사이킷런으로 훈련 세트와 테스트 세트 나누기
* $train\_test\_split()$: 이 함수는 전달되는 리스트나 배열을 비율에 맞게 훈련 세트와 테스트 세트로 나누어 준다. 물론 나누기 전에 알아서 섞어준다.
* $train\_test\_split()$ 함수에도 자체적으로 랜덤 시드를 지정할 수 있는 $random\_state$ 매개변수가 있다.
* 이 함수는 기본적으로 25%를 테스트 세트로 떼어낸다.

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
```



#### 무작위로 데이터를 나누었을 때 <u>샘플이 골고루 섞이지 않는 경우</u>
```python
# 기본값은 25%로 테스트 세트를 구성하는 것이 아닌 클래스 비율에 맞게 데이터를 나눈다.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
```

* 특히 일부 클래스의 개수가 적을 때 이런 일이 생길 수 있다.
* 훈련 세트와 테스트 세트에 샘플의 클래스 비율이 일정하지 않다면 모델이 일부 샘플을 올바르게 학습할 수 없을 것이다.
* train_test_split() 함수의 stratify 매개변수에 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눈다.
* 훈련 데이터가 작거나 특정 클래스의 샘플 개수가 적을 때 특히 유용하다.

### 수상한 도미 한 마리
```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```

```python
print(kn.predict([[25, 150]]))
# [0.]
# 도미(1)가 아닌 빙어(0)로 분류
```

```python
import matplotlib.pyplot as plt
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

distances, indexes = kn.kneighbors([[25, 150]])

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


### 기준을 맞춰라
* 두 특성의 **스케일**이 다르다 ($=$ 두 특성의 값이 놓인 범위가 다르다.)
* 알고리즘이 특히 거리 기반일 경우, 특성 데이터를 표현하는 기준이 다르면 올바르게 예측할 수 없다.
* 이런 알고리즘들은 샘플 간의 거리에 영향을 많이 받으므로 제대로 사용하려면 특성값을 <u>일정한 기준으로 맞춰 주어야 하는데</u>, 이러한 작업을 **데이터 전처리**라고 부른다.

```python
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlim((0, 1000)) # x축의 범위를 (0, 1000)으로 y축과 동일하게 설정
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


* **<u>즉, 모델이 실제로 거리를 측정할 때 거의 무게만을 참조한다는 의미</u>**

### 표준점수(z 점수)
* 가장 널리 사용하는 전처리 방법 중 하나
* <u>표준점수는 각 특성값이 0에서 표준편차의 몇 배만큼 떨어져 있는지를 나타낸다.</u>
* 이를 통해 실제 특성값의 크기와 상관없이 동일한 조건으로 비교할 수 있다.

```python
# 0열과 1열의 데이터 특성이 다르므로
mean = np.mean(train_input, axis=0) 

std = np.std(train_input, axis=0)
```


### 전처리 데이터로 모델 훈련하기

```python
print(mean, std)
# [ 27.29722222 454.09722222] [  9.98244253 323.29893931]

train_scaled = (train_input - mean) / std

new = ([25, 150] - mean) / std # 브로드캐스팅(broad-casting)
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

* 표준점수로 변환하는 전처리과정을 거친 후 *새로운 입력 데이터*에 대해서 산점도상에 표시하려는 경우에는, **훈련 세트의 mean, std를 이용해서 변환해야 한다.**

* **브로드캐스팅**: 크기가 다른 넘파이 배열에서 자동으로 사칙 연산을 모든 행이나 열로 **확장**하여 수행하는 기능이다.


---

```python
kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) / std

kn.score(test_scaled, test_target) # 1.0
print(kn.predict([new])) # [1.]
```

```python
distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')

plt.xlabel('length')
plt.ylabel('weight')

plt.show()
```


### $scikit-learn$
* $train\_test\_split()$: 훈련 데이터를 훈련 세트와 테스트 세트로 나누는 함수이다. 테스트 세트로 나눌 비율은 $test\_size$ 매개변수에서 지정할 수 있으며 기본값은 <b>0.25(25%)</b>이다.
* 위 함수는 $shuffle$ 매개변수로 훈련 세트와 테스트 세트로 나누기 전에 무작위로 섞을 지 여부를 결정할 수 있다.(기본값은 $True$)
* $stratify$ 매개변수에 클래스 레이블이 담긴 배열(일반적으로 **타깃 데이터**)을 전달하면 **클래스 비율에 맞게 훈련 세트와 테스트 세트를 나눈다.**
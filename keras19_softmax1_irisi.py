import numpy as np
from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf


#1. 데이터
datasets = load_iris()
# print(datasets.DESCR) #판다스 describe()
# print(datasets.feature_names)  # 판다스 columes() #Feature ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
# print(x.shape,y.shape) # (150, 4) (150,)
# print(x)
# print(y) #데이터가많아지면,다중분류면 y의 라벨 확인 #컬럼-> 디멘션-> 노드 갯수

#print(x.shape,y.shape)
print(x)
print(y)
print('y의 라벨값:', np.unique(y))
###############################요지점에서 원핫을 해야겠죠?

from keras.utils import to_categorical
y = to_categorical(y)
#print(y)
print(y.shape) 
## y를(150,)-> (150, 3) 판다스.겟더미 # 사이킷에 원핫인코더
###################################################
x_train, x_test, y_train, y_test = train_test_split(x, y, 
     shuffle= True, random_state=942, 
     train_size= 0.8, 
     stratify=y)# random_state=123) #내가 모르면 2를쓰자 데이터가 한쪽으로 치우칠수 있으므로 y라벨값 비율의개수만큼 빼준다.
print(np.unique(y_train,return_counts=True))

#튜닝자동화

# #2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=4, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) #3개노드를 뽑기위해서


# #3 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, verbose=1,)

#accuracy_score 를 사용해서 스코어를 빼세요
# # 4. 평가, 예측


# # 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result: ', result)
y_predict = np.round(model.predict(x_test))
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print(y_predict)
print('acc:', acc)






#3개이상 다중분류할떄 사용하는 함수 Softmax(소프트맥스)는 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수이다.
#※relu : 음수를 0으로 만드는 함수
#시그모이드 : 임의의 값을 [0,1] 사이로 압축한다.
#이진 분류:2가지 선택으로 분류하는거 쓰는 함수  binary_crossentropy sigmoid 최종값=마지막노드의값 
#  y라벨의갯수만큼 노드를잡는다, 
#원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 
#다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식입니다. 이렇게 표현된 벡터를 원-핫 벡터(One-Hot vector)라고 합니다.
# 원핫
#        0 1  2
# 가위 0 1 0  0 =1
# 바위 1 0 1 0 =1
# 보   2 0 0 1=1 라벨의갯수만큼  쉐이프가늘어난다
#원핫 예
# 010 -> 0.1 0.7 0.2  #0  1 0 (O)
# 001 -> 0.5 0.2 0.3  #1 0  0 (x)
#주가예측 시험
#정리
#회귀 ==             == 분류 
#mse,ma. ㅣ이진, 다중
#최종 레이어 회귀는 ㅣ bivervy_crosst ,cateegorical_crosstropy
#          linear  ㅣ  sigmoid softmax
#원핫 x       x     o
#최종layer,노드 갯수,행의컬럼의갯수 ## 1,y의값

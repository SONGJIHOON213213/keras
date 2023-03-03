from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x)
# print(y)

# print(datasets)

# 1~1조를 1로 나오게하려면 최대값1조로나누면된다
#데이터 가 정제 전처리가 되어있는 데이터
#feature 열,특성
print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

print(datasets.DESCR)

print(x.shape, y.shape) #(506, 13) (506,) 인포티메이션 13개 최종 1개 스칼라 506개 벡터 1개

x_train, x_test, y_train, y_test = train_test_split(x,y,
 train_size = 0.75, shuffle=True, random_state=11) # x_train <-x  x_test <-x 먼저 y 동일



####################### [실습]###############
#1. train 0.7
#2. R2 0.8 이상
#############################################

#2. 모델구성
model = Sequential()
model.add(Dense(20,input_dim=10))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(32))
model.add(Dense(20))
model.add(Dense(1))





#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test) #전체 x를예측하려면

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict) #0.99까지올리기
print('r2 스코어 :', r2)


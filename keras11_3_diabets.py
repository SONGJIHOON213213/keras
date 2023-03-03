from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1 데이터.
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
 train_size = 0.62, shuffle=True, random_state=10) 

#[실습]
#R2 0.62 이상 

#2. 모델 구성 
model = Sequential()
model.add(Dense(10,input_dim=10))
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

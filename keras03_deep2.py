#1. 데이터
import numpy as np 
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model = Sequential() 
model.add(Dense(3, input_dim=1)) 
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000) #400번 [[4.0626106]] 1000번 [[4.0000033]]

#4 평가, 예측
loss = model.evaluate(x, y) # 변수 이름은 가독성 좋게
print('loss : ',loss)

result = model.predict([4]) #위에가 꺽세면 아래도 꺽세로 
print("[4]의 예측값 :", result)



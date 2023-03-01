import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,5])

#2. 모델구성
model = Sequential() 
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=90, batch_size=2)

#4 평가, 예측
loss = model.evaluate(x, y)  #[[6.02649]] 0.23 배치1 [[6.0149026]] 0.21 배치2
print('loss : ',loss)

result = model.predict([6]) 
print("[6]의 예측값 :", result)
 #배치사이즈 디폴트값 32
# x 는 3개
# y 는 2개
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1 데이터.
x = np.array([range(10), range(21,31), range(201,211)]) #3행 10열
print(x.shape) #(3,10)
x= x.T #(10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 ]]) #(2,10)

y = y.T #(10,2)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=3)) #10,3이라서
model.add(Dense(5))
model.add(Dense(4))  
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=30, batch_size=3) 

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ' , loss)

result = model.predict([[9, 30, 210]])
print('[[9,30,20]]의 예측값 : ', result)

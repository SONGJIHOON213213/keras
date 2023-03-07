from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x_train = np.array(range(1,17)) #X의쉐이프10개 1~16
y_train = np.array(range(1,17))
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
#실습 :: 잘라봐!!!
x_val = x_train[14:]
y_val = y_train[14:]

x_test = x_train[11:14]
y_test = y_train[11:14]

#2. 모델
model = Sequential()
model.add(Dense(5,activation='linear', input_dim =1)) #5개면 1~5개 
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=1005,batch_size=4,
          validation_data=(x_val, y_val))
 
 #4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)
 
result = model.predict([17])
print('17의 예측값: ', result)
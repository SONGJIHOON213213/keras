# x 는 3개
# y 는 2개
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1 데이터.
x = np.array([range(10)]) #3행 10열
print(x.shape) #(3,10)
x= x.T #(10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 ],
              [9,8,7,6,5,4,3,2,1,0]]) #(3,10)

y = y.T #(10,3)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) #10,3이라서
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))  
model.add(Dense(7))  
model.add(Dense(6))  
model.add(Dense(4))  
model.add(Dense(3))  
model.add(Dense(2))    
model.add(Dense(3))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=400, batch_size=4) 

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ' , loss)
#[[9,30,20]]의 예측값 10, 1.9, 0
result = model.predict([[9]])
print('[[9]]의 예측값 : ', result)

# [[9,30,20]]의 예측값 :  [[9.825916 1.78858  0.806261]]
# [[9]]의 예측값 :  [[9.939272   1.5774661  0.70911044]]
# [9]의 예측값 : [[10.004706    1.9125221  -0.15821794]]
# [[9]]의 예측값 :  [[10.061804    1.3411418   0.06890403]]
# [[9]]의 예측값 :  [[10.0189295   1.822108    0.01423216]]

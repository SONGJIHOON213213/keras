import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 행무시 열우선 #2가지의 특성을 가지고 하나의 결과를 찾기 #만약에 10행3열이면 dim = 3 
x = np.array(
   [[1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 1],
    [7, 1.4],
    [8, 1.4],
    [9, 1.6],
    [10, 1.4]]
) 
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape) #열 피쳐 컬럼 동일이야기 #(10,2)
print(y.shape)#(10,)

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=2))#열이 2개라 딤은 2개
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=30, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ' ,loss)

result = model.predict([[10, 1.4]])
print('[[10 1.4]]의 예측값 :', result)
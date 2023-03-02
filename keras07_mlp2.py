import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 행무시 열우선 

x = np.array(
    [[1,2,3,4,5,6,7,8,9,10], #2,10
    [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]
) 

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) 

x = x.T #행열 바꾸기 


# (2,10)을 (10,2)로 바꾸세요.

print(x.shape) # (10,2)
print(y.shape) # (10,)
#'''블럭주석


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Dense(5))
model.add(Dense(4))  
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=30, batch_size=5) 

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ' , loss)

result = model.predict([[10, 1.4]])
print('[[10 1.4]]의 예측값 :', result)
"""



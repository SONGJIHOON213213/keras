import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 19])
x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
y = np.array([6,7,8,9,10])

print(x.shape, y.shape)  # (7, 3)(7,)
x = x.reshape(5, 5, 1)  # [[1],[2],[3]],[[2],[3],[4], ........]
print(x.shape)

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(700,activation='relu', input_shape = (5,1)))
model.add(Dense(400, activation ='relu'))
model.add(Dense(200, activation ='relu'))
model.add(Dense(100, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(1,  activation = 'linear'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=600)

# 4. 평가 예측
loss = model.evaluate(x, y)
x_predict = np.array([[6,7,8,9,10]]).reshape(1,5,1)
print(x_predict.shape)

result = model.predict(x_predict)
print('loss: ', loss)
print('[8,9,10]의 결과', result)

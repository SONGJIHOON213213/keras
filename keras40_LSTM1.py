import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM

# 1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 19])
x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
y = np.array([6,7,8,9,10])

print(x.shape, y.shape)  # (7, 3)(7,)
x = x.reshape(5, 5, 1)  # [[1],[2],[3]],[[2],[3],[4], ........]
print(x.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(650), input_shape=(5, 1))
model.add(Dense(60, activation ='relu'))
model.add(Dense(40, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(1,  activation = 'linear'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

# 4. 평가 예측
loss = model.evaluate(x, y)
x_predict = np.array([[6,7,8,9,10]]).reshape(1,5,1)
print(x_predict.shape)

result = model.predict(x_predict)
print('loss: ', loss)
print('[8,9,10]의 결과', result)
#LSTM 4배나온이유 4개의 상호작용으로 120 * 4 = 480
# [8,9,10]의 결과 [[11.034578]]
# [8,9,10]의 결과 [[10.940032]]
# [8,9,10]의 결과 [[10.914484]]
# [8,9,10]의 결과 [[10.963728]] 
#  loss:  0.003114940132945776
#  [8,9,10]의 결과 [[11.050832]]
# [[11.160572]]
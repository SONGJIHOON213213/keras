import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# y = ?
x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],
              [5,6,7,8,9]])

y = np.array([4,5,6,7,8,9,10])

print(x.shape,y.shape) #(7,3)(7,)
#x의 shape =(행,열,몇개씩 훈련하는지!!)

x = x.reshape(5,5,1) #[[1],[2],[3]],[[2],[3],[4], ........]
print(x.shape)

#2.모델구성
model = Sequential()
model.add(SimpleRNN(64, input_shape = (5,1)))
model.add(Dense(64, activation ='relu'))
model.add(Dense(32, activation ='relu'))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4.평가예측
loss = model.evaluate(x, y)
x_predict = np.array([7,8,9,10]).reshape(1,4,1) #[[8],[9],[10]]]1차원이라 3차원으로 바꾸기위해서 reshape
print(x_predict.shape) 

result = model.predict(x_predict)
print('loss: ',loss)
print('[6,7,8,9,10]의결과',result)

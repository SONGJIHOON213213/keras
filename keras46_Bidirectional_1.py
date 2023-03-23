import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,GRU,LSTM,Bidirectional
# 1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 19])
x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
y = np.array([6,7,8,9,10])

print(x.shape, y.shape)  # (7, 3)(7,)
x = x.reshape(5, 5, 1)  # [[1],[2],[3]],[[2],[3],[4], ........]
print(x.shape)

 #2. 모델구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(10),return_sequences=True), input_shape=(3,1))
model.add(Dense(1))
model.add(LSTM(10))
model.add(Bidirectional(GRU(10)))

model.summary()

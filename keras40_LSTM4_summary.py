import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM, GRU

#1.데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 19])
x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
y = np.array([6,7,8,9,10])
print(x.shape,y.shape) #(7,3)(7,)
# x의 shape =(행,열,몇개씩 훈련하는지!!)

x = x.reshape(5,5,1) #[[1],[2],[3]],[[2],[3],[4], ........]
print(x.shape)

#2.모델구성
model = Sequential() #[batch행의크기,timesteps열,feature 훈련]
model.add(LSTM(10,input_shape = (5,1)))
# # units * (feature + bias + units) = 파람수
model.summary()
# model.add(Dense(7, activation ='relu'))
# model.add(Dense(1))
# model.summary()

# #3.컴파일,훈련
# model.compile(loss='mse',optimizer='adam')
# import time 
# start = time.time()
# model.fit(x,y, epochs=100)
# end = time.time()

# #4.평가예측
# loss = model.evaluate(x, y)
# x_predict = np.array([6,7,8,9,10]).reshape(1,5,1) #[[8],[9],[10]]]1차원이라 3차원으로 바꾸기위해서 reshape
# print(x_predict.shape) 

# result = model.predict(x_predict)
# print('loss: ',loss)
# print('[6,7,8,9,10]의결과',result)
# print("걸린시간: ", round(end - start))



#20 + 460 + 20 = 480
#4배

#  Layer (type)                Output Shape              Param #
# =================================================================
#  gru (GRU)                   (None, 10)                390

# =================================================================
# Total params: 390
# Trainable params: 390
# Non-trainable params: 0
# _________________________________________________________________ 

#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 10)                480

# =================================================================
# Total params: 480
# Trainable params: 480
# Non-trainable params: 0
# _________________________________________________________________

#3 * (다음 노드 수^2 +  다음 노드 수 * Shape 의 feature + 다음 노드수 )
# 3 * (10^2 + 1*10 + 10) = 360 의 값이 나오게 됩니다.
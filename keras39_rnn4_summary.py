import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1.데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 19])
x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
y = np.array([6,7,8,9,10])
print(x.shape,y.shape) #(7,3)(7,)
#x의 shape =(행,열,몇개씩 훈련하는지!!)

x = x.reshape(5,5,1) #[[1],[2],[3]],[[2],[3],[4], ........]
print(x.shape)

#2.모델구성
model = Sequential() #[batch행의크기,timesteps열,feature 훈련]
model.add(SimpleRNN(32,input_shape = (5,1)))
# units * (feature + bias + units) = 파람수

model.add(Dense(32, activation ='relu'))
model.add(Dense(1))
model.summary()

# #3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
import time 
start = time.time()
model.fit(x,y, epochs=100)
end = time.time()


# #4.평가예측
loss = model.evaluate(x, y)
x_predict = np.array([6,7,8,9,10]).reshape(1,5,1) #[[8],[9],[10]]]1차원이라 3차원으로 바꾸기위해서 reshape
print(x_predict.shape) 

result = model.predict(x_predict)
print('loss: ',loss)
print('[6,7,8,9,10]의결과',result)
print("걸린시간: ", round(end - start))
# ====================================================================
# Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 32)                1088

#  dense (Dense)               (None, 1)                 33

# ===================================================================
# Total params: 1,121
# Trainable params: 1,121
# Non-trainable params: 0
# _________________________________________________________________

#(input_size, output_size) = (1, 32)
#(output_size, output_size) = (32, 32)
#따라서 이 레이어의 총 파라미터 수는 32 + 1024 + 32 = 1088입니다.
#=====================================================================
#cmd 명령어정리
#cmd cls 치면 클리어
#conda -create -n tf273cpu python=3.9.13 anaconda
# 
# 댄스,2~3~4~5차원다먹힘, 
# RNN은 3차원 
# 
# 
# #===================================================================
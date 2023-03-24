import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,GRU,Bidirectional,LSTM,Conv1D,Flatten
#1.데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70]) # 아워너80
print(x.shape, y.shape)  # (7, 3)(7,)
x = x.reshape(13, 3, 1)  # [[1],[2],[3]],[[2],[3],[4], ........]
print(x.shape)

#2.모델구성
model = Sequential()
#model.add(LSTM(10,input_shape =(3,1)))     #토탈파람스 : 541
model.add(Conv1D(10,2, input_shape = (3,1)))#Conv에선필터10개lstm에선 유닛 커널사이즈 #토탈파람스 : 141 
model.add(Conv1D(10,2))                     #Conv에선필터10개lstm에선 유닛 커널사이즈 #토탈파람스 : 301 
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))
model.summary()


#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
import time 
start = time.time()
model.fit(x,y, epochs=2000)
end = time.time()

#4.평가예측
loss = model.evaluate(x, y)
x_predict = np.array([50,60,70]).reshape(1,3,1) #[[8],[9],[10]]]1차원이라 3차원으로 바꾸기위해서 reshape
print(x_predict.shape) 

result = model.predict(x_predict)
print('loss: ',loss)
print('[50,60,70]의결과 80이상 ',result)
print("걸린시간: ", round(end - start))

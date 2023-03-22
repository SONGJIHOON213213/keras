import numpy as np
from sklearn.metrics import r2_score
dataset = np.array(range(1,101)) #100개
timesteps = 5
x_test = np.array(range(96,106))


def split_x(dataset, timesteps):

    forlistgen = (dataset[i : i + timesteps] 
    for i in range (len(dataset)-timesteps + 1) )
    return np.array(list(forlistgen))

bbb = split_x(dataset , timesteps)

x = bbb[:, :-1]# : = 자른다는의미
y = bbb[:,-1]

datasetforpred = np.array(range(106))
def split_x(dataset, timesteps):

    forlistgen = (dataset[i : i + timesteps] 
    for i in range (len(dataset)-timesteps + 1) )
    return np.array(list(forlistgen))

x_test=split_x(datasetforpred,4)
y_test=x_test[:,-1]+1
def RNN_reshape(x):
    return np.reshape(x,list(x.shape)+[1])
x = RNN_reshape(x)
x_test = RNN_reshape(x_test)
print(x_test,y_test)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
print(x.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(units = 32, input_shape = (4,1) ,activation = 'linear'))
model.add(Dense(1))


#3.컴파일,훈련
model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y, epochs = 100, batch_size = 30)

#4.평가,예측 
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("100~106까지 값이 나오게 ", r2)
#ctrl + shift + L 한꺼번에 수정
#컨트를 알트 상하방향키 여러줄 수정 
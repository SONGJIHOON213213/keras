import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.utils import to_categorical

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
# def RNN_reshape(x):
#     return np.reshape(x,list(x.shape)+[1])
# x = RNN_reshape(x)
# x_test = RNN_reshape(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(units = 400, input_shape = (4,) ,activation = 'linear'))
model.add(Dense(units = 50,activation = 'linear'))
model.add(Dense(units = 40,activation = 'linear'))
model.add(Dense(1))
model.summary()

#3.컴파일,훈련
model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs=500, batch_size= 80, validation_split = 0.2)


#4.평가,예측 
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("100~106까지 값이 나오게 " , r2)
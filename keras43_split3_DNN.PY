import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
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

train_set = np.array(range(106))

x_test=split_x(train_set,4)
y_test=x_test[:,-1]+1

# a = split_x(x_test, timesteps)
# # print(a)
# a1 = a[: , :-1]
# # print(a1)

# def RNN_reshape(x):
#     return np.reshape(x,list(x.shape)+[1])
# x = RNN_reshape(x)
# x_test = RNN_reshape(x_test)

# 2. 모델 구성 
model = Sequential()
model.add(Dense(units = 560, input_shape = (4,)))
model.add(Dense(units = 40,activation = 'relu'))
model.add(Dense(units = 30))
model.add(Dense(1))
model.summary()

#3.컴파일,훈련
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
model.fit(x , y, epochs=2000, batch_size=64, verbose=1, validation_split=0.2, callbacks=[es])

#4.평가,예측 
result = model.evaluate(x,y)
print('result:', result)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("100~106까지 값이 나오게 " , r2) 
#100~106까지 값이 나오게  0.9998284590527362
#100~106까지 값이 나오게  0.9999991371061387
# #model.fit(x,y,epochs=500, batch_size= 100, validation_split = 0.2)
# LSTM 인풋랭스 
# 시계열데이터는 Y값이 없다. 개발자가 지정해야된다
# 1. 타임스텝,2.ㄱㄱ
# 컬럼이 (13 n,13)
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense,Dropout,LeakyReLU,Conv2D, Flatten, MaxPooling2D,Conv1D,Reshape,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# 0. seed initialization
#랜덤시드 값이 똑같이나오다 몇번돌려도
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
df=pd.read_csv('./keras/_data/jena_climate/jena_climate_2009_2016.csv',index_col=0)
# print(df)
# print(df.columns)
# print(df.info())
# print(df.describe())

print(type(df[df.columns[3]].values))
# import matplotlib.pyplot as plt
# plt.plot(df[df.columns[3]].values)
# plt.show()

x=df.drop(df.columns[3],axis=1)
y=df[df.columns[3]]

ts=6
#70:20:10 0.7 0.3* 2/3
#80:10:10 0.8 0.2* 1/2
#60:30:10 0.6 0.4* (3/4) 
#50:40:10 0.5 0.5* (4/5) 
#50:40:10 0.5 0.5 (4/5)
#50 = 0.5  4/5  <- 40:  4+1(5)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,shuffle=False)
x_test,x_predict,y_test,y_predict=train_test_split(x_test,y_test,train_size=2/3,shuffle=False)

# df_train,df_test,df_train,df_test=train_test_split(df,df,train_size=0.7)

# print(x_train.shape,y_train.shape)
# print(x_test.shape,y_test.shape)
# print(x_predict.shape,y_predict.shape)

def hello_splitx(x,ts,scaler):    # timesplit 는 함수다 내장함수가 아니고 정의한거다
    x=scaler.transform(x)
    gen=(x[i:i+ts] for i in range(len(x)-ts+1))
    return np.array(list(gen))[:,:-1] #: 마지막데이터를안쓰겟다

def hello_splity(y,ts):
    gen=(y[i:i+ts] for i in range(len(y)-ts+1))
    return np.array(list(gen))[:,-1]

scaler=MinMaxScaler()
scaler.fit(x_train)

x_train=hello_splitx(x_train,ts,scaler)
x_test=hello_splitx(x_test,ts,scaler)
x_predict=hello_splitx(x_predict,ts,scaler)

y_train=hello_splity(y_train,ts)
y_test=hello_splity(y_test,ts)
y_predict=hello_splity(y_predict,ts)
#모든 데이터를 받아서 예측하려고 타임스플리트


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
print(x_predict.shape,y_predict.shape)

# (294366, 19, 13) (294366,)
# (84091, 19, 13) (84091,)
# (42037, 19, 13) (42037,)

# 2. model build
model=Sequential()
model.add(LSTM(200,input_shape=(5, 13)))
model.add(Dense(64,activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation= 'relu'))
model.add(Dense(1))
# model.summary()
# (42036, 19, 13) (42036,)
# (42037, 19, 13) (42037,)


# 3. compile,training
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train
          ,epochs=100,batch_size= 1000
          ,validation_split=0.1,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min'
          ,patience=50
          ,restore_best_weights=True,verbose=True))

# 4. predict,evaluate
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print(f'RMSE : {RMSE(y_test,model.predict(x_test))}')
import numpy as np
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense,SimpleRNN,Conv1D,Flatten
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

#1.데이터 
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape) 
print(x_test.shape)

x_train = x_train.reshape(50000,32*3,32)
x_test = x_test.reshape(10000, 32*3,32)
#2.모델
model = Sequential()
model.add(Conv1D(10,2,activation='relu', input_shape = (96,32)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(3))

#3.컴파일,훈련
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5 ,batch_size= 80, validation_split = 0.2)

# #4. 평가, 예측 
model.evaluate(x_test,  y_test, verbose=1)

# 4차원 -> 3차원으로 바꾸려면 50000,32*3,32 식으로 바꿔줘야된다 
# (50000, 32, 32, 3)
# (10000, 32, 32, 3) 4차원에 3개로 쌓여있다.

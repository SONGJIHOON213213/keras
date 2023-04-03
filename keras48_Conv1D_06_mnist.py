import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense ,SimpleRNN,Flatten,Conv1D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) 
print(x_test.shape)
# (60000, 28, 28)
# (60000,)
# x_train = x_train.reshape(60000,28,28)
# x_test = x_test.reshape(10000, 28,28)

# # # 2. 모델 구성
# model = Sequential()
# model.add(Conv1D(10,2,activation='relu', input_shape = (28,28)))
# model.add(Flatten())
# model.add(Dense(16))
# model.add(Dense(16))
# model.add(Dense(16))
# model.add(Dense(1))
# model.summary()


# # #컴파일 훈련
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=10, batch_size = 150,validation_split=0.2)

# #4. 평가, 예측 
# model.evaluate(x_test,  y_test, verbose=1)

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense ,SimpleRNN
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
x_train = x_train.reshape(60000,28,28)
x_test = x_test.reshape(10000, 28,28)

# print(np.unique(y_train))
# for i in range(10):
#     print(y_train[i])
# y_train = to_categorical(y_train)
# # print(y_train.shape)

# y_test = to_categorical(y_test)  #원핫인코딩 숫자배열을 0101 로보기위해서 바꾸는거
# # print(y_test.shape)

# for i in range(10):
#     print(y_train[i])

# print(x_train.shape) 

# # # # #2. 모델구성
model = Sequential()
model.add(SimpleRNN(600,activation='relu', input_shape = (28,28))) #(20640, 8) (20640,)
model.add(Dense(400, activation ='relu'))
model.add(Dense(200, activation ='relu'))
model.add(Dense(100, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(150,  activation = 'linear'))

#1-1스케일링
#scaler : (이미지)0~255사이 => MinMax가 가장 괜찮/ 255로 나누는 경우도 있음 

#1) 이미지 스케일링 방법
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# #print(np.max(x_train), np.min(x_train)) #1.0.0.0

# x_train = x_train.reshape(60000,28*28)/255.0 #reshape, scale 같이 씀
# x_test = x_test.reshape(10000,784)/255.0

# scaler = MinMaxScaler() 
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
# x_test = scaler.transform(x_test)

# #컴파일 훈련
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size = 150,validation_split=0.2)

#4. 평가, 예측 
model.evaluate(x_test,  y_test, verbose=1)

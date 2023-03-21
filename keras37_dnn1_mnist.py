import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder
#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) 
print(y_train.shape)
# (60000, 28, 28)
# (60000,)
x_train= np.reshape(x_train,(x_train.shape[0],-1))
x_test = np.reshape(x_test,(x_test.shape[0],-1))

print(np.unique(y_train))
for i in range(10):
    print(y_train[i])
y_train = to_categorical(y_train)
# print(y_train.shape)

y_test = to_categorical(y_test)  #원핫인코딩 숫자배열을 0101 로보기위해서 바꾸는거
# print(y_test.shape)

for i in range(10):
    print(y_train[i])

print(x_train.shape) 

#2.모델구성
model = Sequential()
model.add(Dense(32, input_shape = (x_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#3. 컴파일, 훈련 
model.fit(x_train, y_train, epochs=10, batch_size = 64,validation_split=0.2)

#4. 평가, 예측 
model.evaluate(x_test,  y_test, verbose=1)
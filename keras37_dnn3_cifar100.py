import numpy as np
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder

#1.데이터 
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape) 
print(x_test.shape)

x_train= np.reshape(x_train,(50000,32*32*3))
x_test = np.reshape(x_test,(10000,32*32*3))
print(x_train.shape) 

print(np.unique(y_train))
y_train = to_categorical(y_train)
print(y_train.shape)

y_test = to_categorical(y_test)
print(y_test.shape)

print(x_train.shape) 

model = Sequential()
model.add(Dense(80, input_shape = (3072,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(100, activation='softmax'))
# ValueError: Shapes (80, 100) and (80, 10) are incompatible 80~100 으로설정
#3.컴파일,훈련
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=50,batch_size= 80, validation_split = 0.2)

# #4. 평가, 예측 
model.evaluate(x_test,  y_test, verbose=1)



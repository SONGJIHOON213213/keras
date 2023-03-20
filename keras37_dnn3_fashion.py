import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
#1.데이터
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
print(x_train.shape)
print(x_test.shape)
# (60000, 28, 28)
# (10000, 28, 28)

x_train = np.reshape(x_train,(60000,28*28))
x_test = np.reshape(x_test,(10000,28*28))
print(x_train.shape)

print(np.unique(y_train))
y_train = to_categorical(y_train)
print(y_train.shape)

y_test = to_categorical(y_test)
print(y_test.shape)
print(x_train.shape)

model = Sequential()
model.add(Dense(80, input_shape =(784, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3.컴파일,훈련
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=1000,batch_size= 80, validation_split = 0.2)

# #4. 평가, 예측 
result = model.evaluate(x_test, y_test)
print('result:', result)

y_predict = np.round(model.predict(x_test)).astype(int)

acc = accuracy_score(y_test, y_predict)
print('acc:', acc)

#컬럼 784개를 다사용할필요가없다.
#트랜스포머 모델, (시계열 모델은RNN모델을쓴다.)
#(N.13) -> 4차원으로  N(13,1,1)
#캘리포니아 (N,8)  -> (N,8,1,1) / (N,4,2,1) / (N,2,2,2) / (N,2,4,1)


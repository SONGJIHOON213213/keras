from tensorflow.keras.datasets import  fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Conv1D,Reshape,LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) =fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255.
print(x_train.shape,y_train.shape)

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(28,28,1),padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3)))
model.add(Conv2D(10,3))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Reshape(target_shape=(25,10)))
model.add(Conv1D(10,3, padding = 'same'))
model.add(LSTM(784))
model.add(Reshape(target_shape=(28,28,1)))
model.add(Conv2D(32,(3,3),padding = 'same'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.summary()
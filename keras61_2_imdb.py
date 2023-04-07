from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input,Reshape,Embedding
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
(x_train,y_train),(x_test,y_test) = imdb.load_data(
    num_words = 10000
)
print(x_train)
print(y_train)
print(x_train,x_test.shape)
print(np.unique(y_train, return_counts=True))
print(pd.value_counts(y_train))

print("뉴스 기사의 최대길이 : ",max(len(i) for i in x_train))
print("뉴스 기사의 평균길이 : ",sum(map(len,x_train))/len(x_train))

maxlen = 100
x_train = pad_sequences(x_train, padding='pre', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='pre', maxlen=maxlen)

print(x_train.shape) #(8982,100)


model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(250))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size= 80, validation_split = 0.2)

acc = model.evaluate(x_test, y_test)[1]
print('acc:',acc)

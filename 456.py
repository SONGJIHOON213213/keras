import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM,Flatten 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split



# Read in data and preprocess
path = 'C:/study/_data/시험/'
x = pd.read_csv(path + 'AIR_HOUR_2021.csv', index_col=0)
y = pd.read_csv(path + 'AIR_HOUR_2022.csv', index_col=0)

print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)




# # 2. 모델구성
model = Sequential()
model.add(Dense(32,input_shape=(6)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax'))

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=200, restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.2, callbacks=[es])


# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print(result)
print('loss',result[0])
print('acc', result[1])

y_predict = model.predict(x_test)
print(y_predict.shape)
y_predict = np.argmax(y_predict, axis=-1)
print(y_predict.shape)
y_true = np.argmax(y_test, axis=-1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_predict)
print('accuracy_score : ', acc)
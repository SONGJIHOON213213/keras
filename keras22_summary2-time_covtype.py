import numpy as np
from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping

#1 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (1797, 64), (1797,)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
print(type(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)

# print(y_train)
print(np.unique(y_train, return_counts=True))


# 2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 32)                1760
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                2112
# _________________________________________________________________
# dense_2 (Dense)              (None, 64)                4160
# _________________________________________________________________
# dense_3 (Dense)              (None, 32)                2080
# _________________________________________________________________
# dense_4 (Dense)              (None, 7)                 231
# =================================================================
# Total params: 10,343

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=10, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)
end_time = time.time()


# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print(result)
print('result',result[0])
print('acc' ,result[1])

print("걸린시간 :" ,round(end_time - start_time),2)
y_predict = model.predict(x_test)
print(y_predict.shape)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict.shape)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_predict)
print('accuracy-_score : ', acc)

#sparse_categorical_crossentropy onehotencoder 안쓸때사용

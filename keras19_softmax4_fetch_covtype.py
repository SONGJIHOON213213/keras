import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping


# 1. 데이터
datasets = fetch_covtype()
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
model.add(Dense(32, input_dim=54, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=200, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.2)

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
print('accuracy-_score : ', acc)


# 카테고리컬 임의로 추가되면 삭제하는 명령어 
# y = to categorical(y)
# y = npdelete(y,0, axis=1)
# print(y.shape)
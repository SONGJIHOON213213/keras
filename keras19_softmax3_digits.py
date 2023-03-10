from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import numpy as np

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (1797, 64), (1797,)

y = to_categorical(y)
print(y.shape)     # (1797, 64), (1797,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)

# print(y_train)
print(np.unique(y_train, return_counts=True))

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

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


# #1.순서

# 1.print('y의 라벨값:', np.unique(y))
#   print(y)
# 2.print('y의 라벨값:', np.unique(y))
# 3.y의라벨이 9개면 model.add(Dense(10, activation='softmax'))
# 4.y의라벨 값

#acc: 0.9583333333333334 
#accuracy 를 쓰기위해 넘파일 round 사용
#np.agmax
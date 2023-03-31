from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Dense, Flatten,MaxPool2D,LeakyReLU
from sklearn.model_selection import train_test_split
import numpy as np
import time
# 1. 데이터
save_path = 'd:/study_data/_save/men_woman/'

men_woman_x_train = np.load(save_path + 'keras56_men_woman_x_train.npy')
men_woman_x_test = np.load(save_path + 'keras56_men_woman_x_test.npy')
men_woman_y_train = np.load(save_path + 'keras56_men_woman_y_train.npy')
men_woman_y_test = np.load(save_path + 'keras56_men_woman_y_test.npy')


# 2. 모델구성
model = Sequential()
model.add(Conv2D(128,(4,4),padding = 'same', input_shape = (150,150,3),activation= LeakyReLU(0.5)))
model.add(MaxPool2D())
model.add(Conv2D(256,(3,3) ,padding = 'valid', activation= LeakyReLU(0.5)))
model.add(MaxPool2D())
model.add(Conv2D(240,(2,2),padding = 'same', activation= LeakyReLU(0.5)))
model.add(MaxPool2D())
model.add(Conv2D(200,(1,1),padding = 'valid', activation= LeakyReLU(0.5)))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation= LeakyReLU(0.5)))
model.add(Dense(64, activation= LeakyReLU(0.5)))
model.add(Dense(50, activation= LeakyReLU(0.5)))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(men_woman_x_train, men_woman_y_train, epochs=1000, validation_data=(men_woman_x_test, men_woman_y_test))

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.subplot(1, 2, 1)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.grid()
plt.legend()
plt.show()

# 4. 평가, 예측
loss = model.evaluate(men_woman_x_test, men_woman_y_test)
print('loss : ', loss)

y_predict = np.round(model.predict(men_woman_x_test))
from sklearn.metrics import accuracy_score
acc = accuracy_score(men_woman_y_test, y_predict)
print('acc : ', acc)
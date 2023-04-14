from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras import regularizers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 1. 데이터
path_save = 'd:/study_data/_save/감정/'

emotion_x = np.load(path_save + 'emotion_x_train.npy')
emotion_y = np.load(path_save + 'emotion_y_train.npy')

emotion_x_train, emotion_x_val, emotion_y_train, emotion_y_val = train_test_split(emotion_x, emotion_y, train_size=0.7, shuffle=True, random_state=123)
# 2. 모델
model = Sequential()
model.add(Conv2D(30, 2, input_shape=(10, 10, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(emotion_x_train, emotion_y_train, epochs=10, batch_size=20, validation_data=(emotion_x_val, emotion_y_val))

# 4. 평가, 예측
loss = model.evaluate(emotion_x_test, emotion_y_test)
print('loss :', loss)

y_pred = model.predict(emotion_x_test)
print('acc : ', accuracy_score(np.argmax(emotion_y_test, axis=1), np.argmax(y_pred, axis=1)))
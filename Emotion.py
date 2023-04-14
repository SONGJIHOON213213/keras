from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import random
from sklearn.metrics import accuracy_score

def emotion(x):
    y=[]
    if type(x) is int:
        x = [x]
    if len(x.shape) == 1:
        for i in range(len(x)):
            if x[i]==0:
                y.append('angry')
            elif x[i]==1:
                y.append('happy')
            elif x[i]==2:
                y.append('sad')
    else:
        for i in range(x.shape[0]):
            if x[i]==0:
                y.append('angry')
            elif x[i]==1:
                y.append('happy')
            elif x[i]==2:
                y.append('sad')
    return y


# def emotion(x):
#     y=[]
#     for i in range(x.shape[0]):
#         if x[i]==0:
#             y.append('angry')
#         elif x[i]==1:
#             y.append('happy')
#         elif x[i]==2:
#             y.append('sad')
#     return y 

# x = np.array([0, 1, 2, 1, 0, 2])
# y = emotion(x)
# print(y)

tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)


# 데이터셋 로드 및 전처리
train_set = 'D:/study_data/_data/color/'


x = train_set
y = test_set
target = 200
batch_size = 5
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('D:/study_data/_data/color/', target_size=(target, target), batch_size=5, class_mode='categorical', color_mode='rgb', shuffle=True)
action = train_datagen.flow_from_directory('D:/study_data/_data/얼굴/', target_size=(target, target), batch_size=5, class_mode='categorical', color_mode='grayscale', shuffle=True)
# face = train_datagen.flow_from_directory('D:/study_data/감정/train/',
#     face, target_size = (target, target), batch_size = 5, class_mode = 'categorical', color_mode = 'grayscale', shuffle = True)

# action = train_datagen.flow_from_directory('D:/study_data/감정/train/',
#     action, target_size = (target, target), batch_size = 5, class_mode = 'categorical', color_mode = 'grayscale', shuffle = True)
x_train = np.concatenate([train_set[0][0], action[0][0]])
y_train = np.concatenate([train_set[0][1], action[0][1]])
x_test = np.concatenate([train_set[1][0], action[1][0]])
y_test = np.concatenate([train_set[1][1], action[1][1]])

print(x_test[0])
print(x_test[0].shape)
print(y_test[0])
print(y_test[0].shape)

# 모델 구성
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(target, target, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()

# 모델 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
es = EarlyStopping(monitor = 'val_accuracy',
                   patience = 30,
                   mode = 'max',
                   verbose = 1,
                   restore_best_weights = True)
# 모델 학습
model.fit(x_train, y_train,
          epochs = 100,
          batch_size = 64,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

a = np.random.randint(0, batch_size)

y_predict = np.argmax(model.predict(x_test[a].reshape(1, target, target, 1)), axis=1)
print('실제 감정 : ', emotion(np.argmax(y_test[a], axis=0)), '\n예측한 감정 : ', emotion(y_predict))



pred_path = 'd:/_study/_data/감정3'
predict = train_datagen.flow_from_directory(pred_path, target_size = (target, target), batch_size = batch_size, class_mode = 'categorical', color_mode = 'grayscale', shuffle = False)

x_pred = predict[0][0]
answer = predict[0][1]

y_pred = np.argmax(model.predict(x_pred), axis=1)

print('실제 사진 예측 색 : ', color(np.argmax(answer, axis=1)), '\n 예측한 색깔 : ', color(y_pred))

print('acc : ', accuracy_score(np.argmax(answer, axis=1), y_pred))
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization,Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,MaxPool2D,Dropout
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
physical_devices = tf.config.list_physical_devices('GPU') #gpu상태
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e) 

#1. 데이터
save_path = 'd:/study_data/_save/train3/'

x_train = np.load(save_path + 'x_train.npy')
x_test = np.load(save_path + 'x_test.npy')
y_train = np.load(save_path + 'y_train.npy')
y_test = np.load(save_path + 'y_test.npy')

#2.모델구성
#2.모델

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(128, 128, 1), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='sigmoid'))


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input(shape=(128,128,1)),
#     tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(1000, activation='relu'),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(500,  activation='relu'),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model = Sequential([
#     Input(shape=(128, 128, 1)),
#     Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dropout(0.3),
#     Dense(1000, activation='relu'),
#     Dropout(0.3),
#     Dense(500, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')
# ])

# model.summary()
model = tf.keras.models.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(128,128,1)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),
    Dropout(0.25),

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),

    MaxPool2D(),
    Flatten(),
    Flatten(),
    Dropout(0.25),
    Dense(2,activation='sigmoid')
])


# 3.컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=80)
hist = model.fit(x_train, y_train, epochs = 1000, batch_size= 40
                 , validation_data=(x_test, y_test),
                 callbacks = [es])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=50)
hist = model.fit(x_train, y_train, epochs = 1000, batch_size= 20
                 , validation_data=(x_test, y_test),
                 callbacks = [es])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs = 1000, batch_size= 40, validation_data=(x_test, y_test))


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

datagen = ImageDataGenerator()
pred_path = 'd:/study_data/_data/train_face/' 
predict = datagen.flow_from_directory(pred_path, target_size=(128,128), class_mode='binary', color_mode='grayscale', shuffle=False)

x_pred = predict[0][0]
y_true = predict[0][1]
# y_true = predict[0][1][:, 1]
y_pred = (model.predict(x_pred)[:, 0] < 0.5).astype(int)

print('실제 범죄자 :', y_true)
print('예측한 범죄자 : ', y_pred)
print('accuracy : ', accuracy_score(y_true, y_pred)) 



# # loss 그래프
# plt.plot(hist.history['loss'], label='train_loss')
# plt.plot(hist.history['val_loss'], label='val_loss')
# plt.title('Model Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # accuracy 그래프
# plt.plot(hist.history['acc'], label='train_acc')
# plt.plot(hist.history['val_acc'], label='val_acc')
# plt.title('Model Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()



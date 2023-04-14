import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization,Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from sklearn.metrics import accuracy_score
physical_devices = tf.config.list_physical_devices('GPU') #gpu상태
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e) 

#1. 데이터
save_path = 'd:/study_data/_save/train/'

x_train = np.load(save_path + 'x_train.npy')
x_test = np.load(save_path + 'x_test.npy')
y_train = np.load(save_path + 'y_train.npy')
y_test = np.load(save_path + 'y_test.npy')

#2.모델구성
#2.모델
model = Sequential()
model.add(Input(shape=(128,128,1)))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2, activation='sigmoid'))


# 3.컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=80)
hist = model.fit(x_train, y_train, epochs = 500, batch_size= 20
                 , validation_data=(x_test, y_test),
                 callbacks = [es])

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
y_true = predict[0][1][:, 1]
y_pred = (model.predict(x_pred)[:, 0] < 0.5).astype(int)

print('실제 범죄자 :', y_true)
print('예측한 범죄자 : ', y_pred)
print('accuracy : ', accuracy_score(y_true, y_pred))



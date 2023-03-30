import numpy as np
from tensorflow.keras.preprocessing import image #전처리preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator #전처리preprocessing
#1.데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,)# 0~1 사이로 나눈다는거는 정규화 한다는 소리 노멀라이제이션,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode = 'nearest') #데이터증폭내용 

#테스트데이터는 평가데이터라서 증폭할필요가없다.  
    
test_datagen = ImageDataGenerator(
    rescale=1./255,    
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',
    target_size=(100,100),  #200에 200으로 확대또는축소
    batch_size=5,  
    class_mode='binary', #0바이너리 넣으면 0과1만됨
    color_mode='grayscale',
    # color_mode='rgb',
    shuffle=True,
)
xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(100,100),  #200에 200으로 확대또는축소
    batch_size=5, #전체 데이터 갯수
    class_mode='binary', #0바이너리 넣으면 0과1만됨
    color_mode='grayscale',    
    # color_mode='rgb',
    shuffle=True,
)
print(type(xy_train))
print(type(xy_train[0]))
print(type(xy_train[0][0]))
print(type(xy_train[0][1]))

#현재 x는 (5,200,200) 짜리 데이터가 32덩어리

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model = Sequential()
model.add(Conv2D(128,(4,4),padding = 'same', input_shape = (100,100,1),activation= LeakyReLU(0.5)))
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

#3.컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['acc'])
hist = model.fit_generator(xy_train, epochs=1000 ,steps_per_epoch = 32 , validation_data=xy_test, validation_steps=24,)   

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('acc: ', acc[-1])
print('val_acc: ', val_acc[-1])
#전체데이터/batch = 160/5 = 32
#발리데이터/batch =  120/5 = 5

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(hist.history['loss'],)
plt.plot(hist.history['val_loss'],)
plt.subplot(1,2,2)
plt.plot(hist.history['acc'],)
plt.plot(hist.history['val_acc'],)
plt.show()
# loss:  4.621817928374128e-10
# val_loss:  0.020789721980690956
# acc:  1.0
# val_acc:  0.9916666746139526

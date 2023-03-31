import numpy as np
from tensorflow.keras.preprocessing import image #전처리preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator #전처리preprocessing
#1.데이터


path = 'd:/study_data/_save/_npy/'

x_train = np.load(path + 'keras55_1_train_x.npy' )
x_test = np.load(path + 'keras55_1_test_x.npy' )
y_train = np.load(path + 'keras55_1_train_y.npy' )
y_test = np.load(path + 'keras55_1_test_y.npy' )

print(x_train)   #(160,100,100,1) (120,100,100,1)
print(x_train.shape)




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
model.add(Dense(1, activation='softmax'))#1->2

#3.컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['acc'])
# hist = model.fit_generator(xy_train, epochs=1000 ,steps_per_epoch = 32 , validation_data=xy_test, validation_steps=24,)   
# hist = model.fit(xy_train[0][0],xy_train[0][1],epochs=10,batch_size =16,validation=(xy_test[0][0],xy_test[0][1]))
hist = model.fit(x_test,y_test, epochs=30 ,
                 steps_per_epoch = 32 , 
                 validation_data=x_test,
                 validation_steps=24
)   
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

# import matplotlib.pyplot as plt
# plt.subplot(1,2,1)
# plt.plot(hist.history['loss'],)
# plt.plot(hist.history['val_loss'],)
# plt.subplot(1,2,2)
# plt.plot(hist.history['acc'],)
# plt.plot(hist.history['val_acc'],)
# plt.show()

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input
import numpy as np 
np.random.seed(0)
import time 
start_time = time.time()
print("시작")
(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
      rescale = 1./255,
      horizontal_flip=True,
    #   vertical_flip=True,
      width_shift_range=0.1,
      height_shift_range=0.1,
      rotation_range=5,
      zoom_range=0.1,
      shear_range=0.1,
      fill_mode='nearest'
      )


augment_size = 100 #10만개 할꺼면 증폭 4만개

# randidx = np.random.randint(60000,size = 40000) #스칼라4만개 벡터 한개
randidx = np.random.randint(x_train.shape[0],size = augment_size) #x_shape 60000,28,28 # [44520 58362 23335 ... 27241 51506 15046]
print(randidx)
print(randidx.shape)
print(np.min(randidx),np.max(randidx)) # 1,59998 애가 변환되면 원값도 바뀜


x_augmented = x_train[randidx].copy() #새로운 데이터 생성
y_augumentd = y_train[randidx].copy() #<- 이렇게 복사하면 문제, 증폭은4차원
print(x_augmented)
print(x_augmented.shape, y_augumentd.shape) # (40000, 28, 28) (40000,)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],1) #전체 데이터 갯수 알떄마 가능 모르면 -1사용
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)


# x_augmented = train_datagen.flow(
#     x_augmented,y_augumentd,batch_size=augment_size,shuffle=False
# )

# print(x_augmented)

# print(x_augmented[0][0].shape) #(40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented,y_augumentd,batch_size=augment_size,shuffle=False
    ,save_to_dir= 'd:/temp/'
).next()[0] #w절대경로,상대경로
end_time = time.time() - start_time
print(augment_size, " 개 증폭에 걸린시간 : ", round(end_time, 2), '초')
print(x_augmented)
print(x_augmented.shape) #(40000,28,28,1)
x_train = np.concatenate((x_train/255. ,x_augmented), axis=0) # (100000, 28, 28, 1)
y_train = np.concatenate((y_train, y_augumentd), axis=0) 
x_test = x_test/255.

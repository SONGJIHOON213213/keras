from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(7, (2,2), padding='same', input_shape=(8,8,1)))        # 출력 : (N, 7, 7, 7)
# (batch_size, rows, columns, channels)

# (2*2)*1*7 +7 =35

model.add(Conv2D(filters=4, kernel_size=(3,), padding='same', activation='relu'))         # 출력 : (N, 5, 5, 4)
# (3*3)*7*4+4 =256

model.add(Conv2D(10, (2,2)))         # 출력 : (N, 4, 4, 10)
# (2*2)*4*10 + 10 =170

model.add(Flatten())            # 출력 : (N, 160)

model.add(Dense(32, activation='relu'))
# (160+1)*32

model.add(Dense(10, activation='relu'))
# (32+1)*10

model.add(Dense(3, activation='softmax'))
# (10+1)*3
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 8, 8, 7)           35          원래는 (N, 7, 7, 7)이지만 패딩으로 인해 모양이 변함.
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 6, 6, 4)           256
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 5, 5, 10)          170
# _________________________________________________________________
# flatten (Flatten)            (None, 250)               0
# _________________________________________________________________
# dense (Dense)                (None, 32)                8032
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                330
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 33
# =================================================================
# Total params: 8,856
# Trainable params: 8,856
# Non-trainable params: 0
# _________________________________________________________________

# 커널사이즈 - 1 = 줄어든 shape
#이게 많아지면 데이터가 소멸됨. 그래서 나온 해결책이 패딩(padding)
#ex) 5*5에서 6*6으로 만듬. 패딩으로 늘어난 값은 0, 결과치에 영향을 주지않음.
#즉, shape가 줄어들지 않음.(이것도 튜닝.)
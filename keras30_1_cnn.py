from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential() #자르는크기
model.add(Conv2D(7,(2,2), 
                    input_shape=(8,8,1))) #출력 : (N,7,7,7) 8-1 8-1 1 [7: (4,4)를 7장으로 늘림]
                       #(batch_size,rows,columns,channels)
model.add(Conv2D(filters=4,kernel_size = (3,3), activation= 'relu'))             #출력 : (N,5,5,4)
model.add(Conv2D(10,(2,2)))#출력 : (N,4,4,13)
model.add(Flatten()) #출력 : (N, 4*4*10) -> (N, 160)
model.add(Dense(32, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))
model.summary()
                                      
                                      
#                                       Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 7, 7, 7)           35
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 5, 5, 4)           256
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 4, 10)          170
# _________________________________________________________________
# flatten (Flatten)            (None, 160)               0
# _________________________________________________________________
# dense (Dense)                (None, 32)                5152
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                330
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 33
# =================================================================
# 채널크기 x 입력 필터 x 출력필터  + 필터의 갯수 =
# 4 x 7  x 1 + 7 = 35
# 9 x 7 x 4 + 4 =256
# 4 x 10 x 4 +10

 # 0,0,0,0   0,0   1,1
#  1,2,3,0   2,3   1,1  0 ex행렬 (2,3) (28,28)->커널사이트1로하면 27,27
#  0,2,3,0                       (2,3)             
#  0,0,1,0
# = 0x1+0x1 +2x1+3x1 + 0x0+0x1 +2x1 3x1
# 높은숫자를 빼서 연산량을 줄이고 값을 높히기 위해 #제일 높은값을 뽑는거 맥스풀링이라고한다 수치만 가져오는거라 연산량0다 그래서 속도가 빠르다. 성능도좋다.
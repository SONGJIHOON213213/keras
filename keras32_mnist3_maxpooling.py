from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping 
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

############################실습############################

# from tensorflow.keras.utils import to_categorical #tensorflow 빼도 가능.
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# Reshape the data
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_train)


from tensorflow.keras.utils import to_categorical #tensorflow 빼도 가능.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#데이터나 순서가 바뀐게 없음. 
#데이터의 구조를 바꾸는 거지 순서, 내용(값)은 바뀌지 않는다.
# (60000, 28, 14, 2)도 가능 14X2이니깐. input shape를 맞출때 사용. 
# ((60000, 28, 7, 2, 2)) 도가능. shape값만 같으면 됨. 
#cf) transpose 랑은 다름. (얘는 내용이 아예바뀜.)

#2. 모델링

model = Sequential()
model.add(Conv2D(64, (2,2), #필터값64
                 padding='same', 
                 input_shape = (28,28,1))) 
model.add(MaxPooling2D())#맥스풀링은 무조건 건너띄게 중첩되지않게 설계되있다. 4x4 -> 2x2
model.add(Conv2D(filters= 64,            
                 kernel_size =(2,2),
                 padding= 'valid', 
                 activation= 'relu'))
model.add(Conv2D(32, 2)) #-< 커널사이트 2,2 3으로쓰면 3,3이란소리
model.add(Flatten())                  
model.add(Dense(10, activation= 'softmax'))                         
model.summary()

# model.add(Dense(15))
# model.add(Dense(10))
# model.add(Dense(10, activation= 'softmax')) #0부터 9까지 여서. print(np.unique(y_train, return_counts = True))
# model.summary()
#원핫 인코더가 필요가 없음. 차원이 이미 갖추어져 있기 때문에

# One-hot encode the target labels



#3. 컴파일 훈련
es = EarlyStopping(monitor= 'val_acc', patience= 15, mode = 'max',
                   restore_best_weights= True,
                   verbose= 1)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 10, 
          batch_size = 3000, verbose = 1,
          validation_split= 0.2,
          callbacks = [es])
results = model.evaluate(x_test,y_test)
print('results :', results)
y_predict = model.predict(x_test)
y_test = np.argmax(y_test, axis =-1)
y_predict = np.argmax(y_predict, axis =-1)
#print(y_predict)
acc = accuracy_score(y_test, y_predict)
print('Accuary score : ', acc)




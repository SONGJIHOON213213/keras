from tensorflow.keras.layers import Conv2D,Flatten,Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
#1.데이터
datasets = load_iris()

x = datasets.data
y = datasets.target
print(np.unique(y))#0,1,2


x_train,x_test ,y_train,y_test = train_test_split(x,y,train_size=0.7, random_state=995, shuffle=True, stratify =y)#stratify 골고루 섞다

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_train))
# print(np.max(x_train))
# [0 1 2]
# 0.0
# 1.0
# print(x_train.shape)
# print(x_test.shape)
x_train = np.reshape(x_train,(105,4,1,1))
x_test = np.reshape(x_test,(45,4,1,1))
# (105, 4)
# (45, 4) 
# print(x_train.shape)
# print(x_test.shape) 

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=18,
                  kernel_size = (4,1) #ctrl + shift + space 뭐쓸지 표시
                 ,padding = 'same'
                 ,activation = 'relu'
                 , input_shape = (4,1,1)
                 ))
model.add(Flatten())
model.add(Dense(63, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.summary()

#3.컴파일,훈련
model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics = 'acc')
model.fit(x_train,y_train,epochs = 5,verbose = 1 , validation_split = 0.2,batch_size = 60)

#4.평가,예측
eva=model.evaluate(x_test,y_test) #eva를 쓰면 acc,loss,가 안에 들어간다.
print('accuracy :',eva[1])
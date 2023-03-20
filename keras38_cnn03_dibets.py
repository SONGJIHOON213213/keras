from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#1 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target
print(x.shape,y.shape) #(442, 10) (442,)
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, random_state=644,train_size=0.7)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.max(x_train)) # 1
# print(np.min(x_train)) # 0
# print(np.max(x_test)) # 1
# print(np.min(x_test)) # 0

# print(np.unique(y)) #원핫인코딩 데이터가 많아서 선택해야되는 모델, 분류모델 개인지고양인지분류,
print(x.shape) #(442, 10)
x_train = np.reshape(x_train,(309,5,2,1)) #4차원 10개자리를 5개 두줄로 놓는방법
x_test = np.reshape(x_test,(133,5,2,1)) #train_set,test_set 모양을 맞춰줘야된다.
# print(x_train.shape) #(309, 10)
# print(x_test.shape) #(133, 10)
#2.모델링
model = Sequential()
model.add(Conv2D(filters=18
                 ,input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])
                 ,kernel_size=(2,2)                 
                 ,padding='same'
                 ,activation='linear'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) #이중 시그모이드 # 다중 소프트맥스

#3.컴파일,훈련 
model.compile(optimizer='adam',loss='mse')
model.fit(x_train,y_train,epochs=5,batch_size = 30,validation_split=0.2)

#4.평가,예측

loss = model.evaluate(x_test,y_test,verbose=1)
print("loss:",loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 score:', r2) 

#'numpy.ndarray' object is not callable 컴파일 구분
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import r2_score, mean_squared_error

#1.데이터 
data_load =  './keras/_data/ddarung/'

train_set = pd.read_csv(data_load+'train.csv'
                                   ,index_col= 0
                        )
test_set = pd.read_csv(data_load + 'test.csv', index_col = 0)
# print(train_set)
# print(test_set)
# print(train_set.isnull().sum())
train_set=train_set.dropna()
# print(train_set.isnull().sum())


x = train_set.drop('count',axis=1)
# print(x)
y = train_set['count']
# print(y)
x_train,x_test ,y_train,y_test = train_test_split(x,y,train_size = 0.7,random_state=952,shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) #(929, 9)
print(x_test.shape)  #(399, 9)

x_train = np.reshape(x_train,(929,3,3,1)) # 9 * 1 * 1 , 3*3*1,3*1*3
x_test = np.reshape(x_test,(399,3,3,1))
print(x_train.shape)
# # print(np.unique(y))
#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=18,
                 kernel_size = (3,3)
                 ,padding = 'same'
                 ,activation = 'relu'
                 ,input_shape = (3,3,1)
                 ))
model.add(Flatten())
model.add(Dense(54, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(1))
model.summary()

#3.컴파일,훈련
model.compile(loss ='mse',optimizer = 'adam')
model.fit(x_train,y_train,epochs = 5, verbose =1 ,validation_split = 0.2, batch_size = 60)

#4 평가, 예측
y_predict= model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("r2,",r2)
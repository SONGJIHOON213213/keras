from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x_train = np.array(range(1,17)) #X의쉐이프10개 1~16
y_train = np.array(range(1,17))
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
#실습 :: 잘라봐!!!
# train_test_split
# 10:3:3


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
train_size=0.625, shuffle=True, random_state=99)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,
train_size=0.5, shuffle=True, random_state=99)

print(x_test, y_test)
print(x_train,y_train)
print(x_val,y_val)



#2. 모델
model = Sequential()
model.add(Dense(5,activation='linear', input_dim =1)) #5개면 1~5개 
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=1005,batch_size=4,
          validation_data=0.2(x_val, y_val))
 
 #4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)
 
result = model.predict([15])
print('17의 예측값: ', result) 
#train_test_split 2개사용
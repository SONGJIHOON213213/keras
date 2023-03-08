from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array(range(1,17)) #X의쉐이프10개 1~16 X의쉐이프 16개
y = np.array(range(1,17))

#실습 :: 잘라봐!!!
# train_test_split
# 10:3:3

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=10/16, random_state=1, shuffle=True)

print(x_train, y_train, x_test, y_test)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=3/6, random_state=123, shuffle=True)

print(x_train, y_train, x_val, y_val, x_test, y_test)


print(x_train.shape,x_test.shape)
print(y_train,y_test.shape)

#2. 모델
model = Sequential()
model.add(Dense(5,activation='linear', input_dim =1)) #5개면 1~5개 
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=10,batch_size=4,validation_split=0.2)#20퍼센트를 발리데이션으로하겟다
 
 #4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)
 
result = model.predict([18])
print('18의 예측값: ', result)

#train_test_split 하나사용

# loss : 66.21041107177734
# 18의 예측값:  [[-3.4704196]]

#1. 데이터
import numpy as np
from sklearn.metrics import r2_score
x1_datasets = np.array([range(100), range(301,401)])
x2_datasets = np.array([range(101,201),range(411,511), range(150,250)]) 

print(x1_datasets.shape)           #(2, 100)  
print(x1_datasets.shape)           #(3, 100)

x1 = np.transpose(x1_datasets) 
x2 = x2_datasets.T             

print(x1.shape)                #(100,2)
print(x2.shape)                #(100,3)

y = np.array(range(2001,2101)) #환율

from sklearn.model_selection import train_test_split #트레인 ,테스트 분리 

x1_train,x1_test,x2_train,x2_test,y_train,y_test = train_test_split(x1,x2,y,train_size= 0.7,random_state=333)

# y_train,y_test = y,train_test_split(y, train_size= 0.7,random_state=333)

print(x1_train.shape,x2_test.shape)
print(x2_train.shape,x2_test.shape)
print(y_train.shape,y_test.shape)

#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,InputLayer,Input
#2-1 모델1
input1 = Input(shape = (2,))
dense1 = Dense(1, activation='relu',name = 'stock1')(input1)
dense2 = Dense(2, activation='relu',name = 'stock2')(dense1)
dense3 = Dense(3, activation='relu',name = 'stock3')(dense2)
output1 = Dense(1, activation='relu',name = 'output1')(dense3)

#2-2.모델2

input2 = Input(shape = (3,))
dense11 = Dense(10,name ='wearther1')(input2)
dense12 = Dense(10,name ='wearther2')(dense11)
dense13=  Dense(10,name ='wearther3')(dense12)
dense14 = Dense(10,name ='wearther4')(dense13)
output2 = Dense(11,name ='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate # 소문자는 함수 대문자는 클래스
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(2, activation = 'relu' , name='mg2')(merge1)
merge3 = Dense(3, activation = 'relu' , name='mg3')(merge2)
last_output = Dense(1,name='last')(merge3)

model = Model(inputs = [input1,input2],outputs=last_output)#2개이상이라 리스트 구성

model.summary()


# wearther1(Dense): 40개의 매개변수, 1000개의 계산
# stock1(Dense): 3개의 매개변수, 200개의 계산
# wearther2(Dense): 110개의 매개변수, 1000개의 계산
# stock2(Dense): 4개의 매개변수, 20개의 계산
# wearther3(Dense): 110개의 매개변수, 1000개의 계산
# stock3(Dense): 9개의 매개변수, 27개의 계산
# wearther4(Dense): 110개의 매개변수, 1000개의 계산
# output1(Dense): 4개의 매개변수, 4개의 계산
# output2(Dense): 121개의 매개변수, 1210개의 계산
# mg1(연결): 추가 매개변수 또는 계산 없음
# mg2(Dense): 39개 매개변수, 39개 계산
# mg3(Dense): 12개 매개변수, 12개 계산
# 마지막(Dense): 4개의 매개변수, 4개의 계산

#3. 컴파일

model.compile(optimizer='adam', loss='mse')
model.fit([x1_train, x2_train], y_train, 
                    validation_data=([x1_test, x2_test], y_test), 
                    epochs=100, batch_size=10)



# # 4. 평가 예측

loss = model.evaluate([x1_test,x2_test],y_test)
print('loss  : ', loss)
y_predict = model.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print('r2 score:',  r2)

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict) 
print('RMSE : ', rmse)
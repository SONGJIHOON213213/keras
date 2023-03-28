#1. 데이터
import numpy as np
from sklearn.metrics import r2_score
x1_datasets = np.array([range(100), range(301,401),range(101,201)])


print(x1_datasets.shape)           
print(x1_datasets.shape)           

x1 = np.transpose(x1_datasets) 
# x2 = x2_datasets.T             
# x3 = x3_datasets.T  

print(x1.shape)                #(100,2)
          #(100,3)

y1 = np.array(range(2001,2101)) #환율
y2 = np.array(range(1001,1101))
from sklearn.model_selection import train_test_split #트레인 ,테스트 분리 

x1_train,x1_test,y1_train,y1_test,y2_train,y2_test = train_test_split(
    x1,y1,y2,train_size= 0.7,random_state=333)


#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,InputLayer,Input
#2-1 모델1
input1 = Input(shape = (3,))
dense1 = Dense(64, activation='relu',name = 'stock1')(input1)
dense2 = Dense(20, activation='relu',name = 'stock2')(dense1)
dense3 = Dense(40, activation='relu',name = 'stock3')(dense2)
output1 = Dense(1, activation='linear',name = 'output1')(dense3)



from tensorflow.keras.layers import concatenate, Concatenate # 소문자는 함수 대문자는 클래스


merge1 = concatenate([output1], name='mg1')
merge2 = Dense(10, activation = 'relu' , name='mg2')(merge1)
merge3 = Dense(30, activation = 'relu' , name='mg3')(merge2)
last_output= Dense(1,name='last')(merge3)

merge11 = Dense(10, activation='relu',name='mg4')(last_output)
merge11 = Dense(10, activation='relu',name='mg4')(merge11)
last_output1= Dense(1,name='last1')(merge2)

last_output2 = Dense(1, name = 'last2')(last_output1)

model = Model(inputs = input1,outputs=[last_output1,last_output2]) 

model.summary()

# 3. 컴파일,훈련
model.compile(loss='mse',optimizer = 'adam')
model.fit(x1_train,[y1_train,y2_train],epochs = 100)
                
# # 4. 평가 예측


result = model.evaluate([x1_test,],[y1_test,y2_test])

y_predict = model.predict([x1_test,])
print(y_predict)
print(len(y_predict), len(y_predict[0]))


from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test,y_predict[0])
r2_2 = r2_score(y1_test,y_predict[1])
print('r2 score:',  (r2_1 + r2_2)/2)


from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y2_test,y_predict))
rmse1 = RMSE(y1_test,y_predict[0])
rmse2 = RMSE(y2_test,y_predict[1])  
print('RMSE1 : ', rmse1 + rmse2/2) 

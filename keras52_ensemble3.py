#1. 데이터
import numpy as np
from sklearn.metrics import r2_score
x1_datasets = np.array([range(100), range(301,401)])
x2_datasets = np.array([range(101,201),range(411,511), range(150,250)]) 
x3_datasets = np.array([range(201,301),range(511,611), range(1300,1400)]) 
print(x1_datasets.shape)           #(2, 100)  
print(x1_datasets.shape)           #(3, 100)

x1 = np.transpose(x1_datasets) 
x2 = x2_datasets.T             
x3 = x3_datasets.T  

print(x1.shape)                #(100,2)
print(x2.shape)                #(100,3)

y1 = np.array(range(2001,2101)) #환율
y2 = np.array(range(1001,1101))
from sklearn.model_selection import train_test_split #트레인 ,테스트 분리 

x1_train,x1_test,x2_train,x2_test,x3_train,x3_test,y1_train,y1_test,y2_train,y2_test = train_test_split(
    x1,x2,x3,y1,y2,train_size= 0.7,random_state=333)



# y_train,y_test = y,train_test_split(y, train_size= 0.7,random_state=333)

# print(x1_train.shape,x2_test.shape,x3_test.shape)
# print(x2_train.shape,x2_test.shape,x3_test.shape)
# print(y_train.shape,y_test.shape,x3_test.shape)

#2.모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,InputLayer,Input
#2-1 모델1
input1 = Input(shape = (2,))
dense1 = Dense(10, activation='relu',name = 'stock1')(input1)
dense2 = Dense(20, activation='relu',name = 'stock2')(dense1)
dense3 = Dense(40, activation='relu',name = 'stock3')(dense2)
output1 = Dense(10, activation='relu',name = 'output1')(dense3)


#2-2.모델2

input2 = Input(shape = (3,))
dense11 = Dense(10,name ='wearther1')(input2)
dense12 = Dense(20,name ='wearther2')(dense11)
dense13=  Dense(40,name ='wearther3')(dense12)
dense14 = Dense(80,name ='wearther4')(dense13)
output2 = Dense(10,name ='output2')(dense14)

#2-2.모델3 

input3 = Input(shape = (3,))
dense11 = Dense(10,name ='wearther5')(input3)
dense12 = Dense(20,name ='wearther6')(dense11)
dense13=  Dense(40,name ='wearther7')(dense12)
dense14 = Dense(80,name ='wearther8')(dense13)
output3 = Dense(10,name ='output3')(dense14)



from tensorflow.keras.layers import concatenate, Concatenate # 소문자는 함수 대문자는 클래스


merge1 = Concatenate(name='mg1',)([output1, output2,output3])
merge2 = Dense(10, activation = 'relu' , name='mg2')(merge1)
merge3 = Dense(30, activation = 'relu' , name='mg3')(merge2)
last_output= Dense(1,name='last')(merge3)

merge11 = Dense(10, activation='relu',name='mg4')(last_output)
merge11 = Dense(10, activation='relu',name='mg4')(merge11)
last_output1= Dense(1,name='last1')(merge2)

last_output2 = Dense(1, name = 'last2')(last_output1)

model = Model(inputs = [input1,input2,input3],outputs=[last_output1,last_output2]) 

model.summary()


# 3. 컴파일,훈련
model.compile(loss='mse',optimizer = 'adam')
model.fit([x1_train,x2_train,x3_train],[y1_train,y2_train],epochs = 100)
                
# # 4. 평가 예측


result = model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])

y_predict = model.predict([x1_test,x2_test,x3_test])
print(y_predict)
print(len(y_predict), len(y_predict[0]))


from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test,y_predict[0])
r2_2 = r2_score(y1_test,y_predict[1])
print('r2 score:',  (r2_1 + r2_2)/2)




from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y1_test,y_predict[0]))
rmse = RMSE(y1_test,y_predict) 
print('RMSE1 : ', rmse) 

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y2_test,y_predict[1]))
rmse = RMSE(y1_test,y_predict) 
print('RMSE1 : ', rmse) 


#==============================================Concatenate,concatenate정리=========================================
# 파이썬에서 "concatenate"는 문자열이나 시퀀스(리스트, 튜플 등)의 요소를 연결하는 것을 의미합니다. 이 작업은 "+" 연산자나 "join()" 메소드를 사용하여 수행할 수 있습니다.
# 그러나 파이썬에서 "Concatenate"와 같이 대문자로 시작하는 것은 기본적으로 정의되어 있지 않은 이름으로 간주됩니다. 
# 따라서 "Concatenate"를 사용하여 문자열이나 시퀀스를 연결하려면 먼저 해당 이름을 정의해야 합니다.
# 예를 들어, 다음은 "concatenate"를 사용하여 두 개의 문자열을 연결하는 방법입니다.
# python
# Copy code
# string1 = "Hello"
# string2 = "world"
# result = string1 + string2
# print(result) # 출력 결과: Helloworld
# 반면에 "Concatenate"를 사용하려면 우선 함수나 메소드를 정의해야 합니다.
# python
# Copy code
# def Concatenate(str1, str2):
# return str1 + str2
# string1 = "Hello"
# string2 = "world"
# result = Concatenate(string1, string2)
# print(result) # 출력 결과: Helloworld
# 따라서 파이썬에서 "concatenate"와 "Concatenate"는 대소문자 구분을 기준으로 서로 다른 의미를 가지며, "concatenate"는 내장 함수나 메소드로 이미 정의되어 있습니다.
# class는 파이썬에서 클래스를 정의할 때 사용하는 키워드입니다. 클래스는 객체 지향 프로그래밍의 핵심 개념 중 하나로, 변수와 메서드를 포함하는 데이터 타입입니다.
# 하지만 []는 파이썬에서 리스트를 생성할 때 사용하는 구문입니다. []는 빈 리스트를 만들 때 주로 사용됩니다.
# 따라서, class = []는 파이썬에서 유효한 구문이 아닙니다. 클래스를 정의할 때에는 class 키워드 뒤에 클래스의 이름을 지정하고, 클래스 내부에 변수와 메서드를 정의해야 합니다.
#==============================================Concatenate,concatenate정리============================================
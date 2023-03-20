from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './keras/_data/kaggle_bike/'   #점 하나 현재폴더의밑에 점하나는 스터디
train_csv = pd.read_csv(path + 'train.csv', 
                        index_col=0) 

print(train_csv)
print(train_csv.shape) #출력결과 (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0) 

                      
print(test_csv)        #캐쥬얼 레지스트 삭제
print(test_csv.shape)  #출력결과 ((6493, 8))
##########################################


print(train_csv.columns) 
# #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
# #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed',]
#       dtype='object')
print(train_csv.info) 

print(type(train_csv)) 

# ################################
# #결측치 처리 1 .제거
print(train_csv.isnull().sum())
train_csv = train_csv.dropna() ####결측치 제거#####
print(train_csv.isnull().sum()) #(11)
print(train_csv.info())
print(train_csv.shape)
# ############################## train_csv 데이터에서 x와y를 분리
x = train_csv.drop(['count','casual','registered'], axis=1) #2개 이상 리스트 
print(x)
y = train_csv['count']
print(y)
# ###############################train_csv 데이터에서 x와y를 분리
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=567
)

scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) #(929, 9)
print(x_test.shape)  #(399, 9)

x_train = np.reshape(x_train,(7620,8,1,1)) # 9 * 1 * 1 , 3*3*1,3*1*3
x_test = np.reshape(x_test,(3266,8,1,1))



#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=18,
                 kernel_size = (1,1)
                 ,padding = 'same'
                 ,activation = 'relu'
                 ,input_shape = (8,1,1)
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
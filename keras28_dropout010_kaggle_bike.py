from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import RobustScaler, MaxAbsScaler 
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd

#1. 데이터
path = './_data/kaggle_bike/'   #점 하나 현재폴더의밑에 점하나는 스터디
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
# # pirnt(train_csv.insul11())
# print(train_csv.isnull().sum())
# train_csv = train_csv.dropna() ####결측치 제거#####
# print(train_csv.isnull().sum()) #(11)
# print(train_csv.info())
# print(train_csv.shape)
# ############################## train_csv 데이터에서 x와y를 분리
# x = train_csv.drop(['count','casual','registered'], axis=1) #2개 이상 리스트 
# print(x)
# y = train_csv['count']
# print(y)
# ###############################train_csv 데이터에서 x와y를 분리
# x_train, x_test, y_train, y_test = train_test_split(
# x, y, shuffle=True, train_size=0.7, random_state=1234567
# )
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

# scaler = MinMaxScaler() # 0.0 711.0 #정규화란, 모든 값을 0~1 사이의 값으로 바꾸는 것이다
# # scaler = StandardScaler() #정답은 (49-50) / 1 = -1이다. 여기서 표준편차란 평균으로부터 얼마나 떨어져있는지를 구한 것이다. 
# # # scaler = MaxAbsScaler #최대절대값과 0이 각각 1, 0이 되도록 스케일링
# # # scaler = RobustScaler #중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
# # x = scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# # x_test = scaler.transform(x_test) 
# # print(np.min(x), np.max(x))




# # # 2. 모델구성
# model = Sequential()
# model.add(Dense(30,input_shape=(13,)))
# model.add(Dropout(0.3))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(0.5))
# model.add(Dense(10))
# model.add(Dropout(0.1))
# model.add(Dense(1))

# # #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=300)
# hist = model.fit(x_train, y_train, epochs=600, batch_size=16, validation_split=0.2, verbose=1, callbacks=[es])

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = np.round(model.predict(x_test))
# print("==============================")
# print(y_test[:5])
# print(y_predict[:5])
# print(np.round(y_predict[:5]))
# print("=============================")
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc:', acc)

y_submit = np.round(model.predict(test_csv))
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['Outcome'] = y_submit
submission.to_csv(path + 'sample_submission36.csv')
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)
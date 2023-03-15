from tensorflow.python.keras.models import Sequential, Model , load_model
import numpy as np
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


# # 0. 시드 초기화
# seed=42
# np.random.seed(seed)
# tf.random.set_seed(seed)


#1. 데이터
path = './_data/wine/'  
train_csv = pd.read_csv(path + 'train.csv', 
                        index_col=0) 

print(train_csv)
print(train_csv.shape) #출력결과 (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0) 
              
print(test_csv)       
print(test_csv.shape)  

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
enc = LabelEncoder()
enc.fit(train_csv['type'])
train_csv['type'] = enc.transform(train_csv['type'])
test_csv['type'] = enc.transform(test_csv['type']) 


x = train_csv.drop(['quality'], axis=1) #2개 이상 리스트 
y = train_csv['quality']
#test_csv =test_csv.drop(['type'],axis =1)
# ###############################train_csv 데이터에서 x와y를 분리

ohe = OneHotEncoder()
y = train_csv['quality'].values
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()


# 1.5 원핫인코딩
# print(np.unique(y))     # [3 4 5 6 7 8 9]
# print(type(y))
# y = pd.get_dummies(y)
# print(type(y))
# print(y)
# y = np.array(y)
# print(type(y))
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=123 #stratify=y
                                                    ) 


6
####scaler = MinMaxScaler() # 0.0 711.0 #정규화란, 모든 값을 0~1 사이의 값으로 바꾸는 것이다
# # # scaler = StandardScaler() #정답은 (49-50) / 1 = -1이다. 여기서 표준편차란 평균으로부터 얼마나 떨어져있는지를 구한 것이다. 
#### scaler = MaxAbsScaler() #최대절대값과 0이 각각 1, 0이 되도록 스케일링
scaler = RobustScaler() #중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv =scaler.transform(test_csv)


# # # # # 2. 모델구성
# model = Sequential()
# dense1 = model.add(Dense(60, input_shape=(12,)))
# dense1 = model.add(Dropout(0.1))
# dense2 = model.add(Dense(50, activation='relu'))# dense2 = model.add(Dropout(0.3))
# dense2 = model.add(Dropout(0.1))
# dense3 = model.add(Dense(40, activation='relu'))
# dense3 = model.add(Dropout(0.1))
# dense4 = model.add(Dense(30, activation='relu'))
# dense4 = model.add(Dropout(0.1))
# dense5 = model.add(Dense(20, activation='relu'))
# dense5 = model.add(Dropout(0.1))
# model.add(Dense(7, activation= 'softmax'))


input1 = Input(shape=(12,))
dense1 = Dense(60, activation='relu')(input1)
drop1 = Dropout(0.1)(dense1)
dense2 = Dense(50, activation='relu')(drop1)
drop2 = Dropout(0.1)(dense2)
dense3 = Dense(40, activation='relu')(drop2)
drop3 = Dropout(0.1)(dense3)
dense4 = Dense(30, activation='relu')(drop2)
drop4 = Dropout(0.1)(dense3)
dense5 = Dense(20, activation='relu')(drop2)
drop5 = Dropout(0.1)(dense4)
output1 = Dense(7, activation='softmax')(drop3)
model = Model(inputs=input1, outputs=output1)
# model = load_model('./_save/keras26_3_save_model.h5')
# model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=1000 #기도메타
                   )

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=60, validation_split=0.2,  callbacks=[es] 
          )

import datetime 
date = datetime.datetime.now()
print(date)
date = date.strftime("%m%d_%H%M%S")
print(date) # 0314_1115
6
# model.save('./_save/keras26_3_save_model.h5')
# 4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result:', result)

y_predict =model.predict(x_test)

y_true = np.argmax(y_test, axis=-1)
y_predict = np.argmax(y_predict, axis=-1)
acc = accuracy_score(y_true, y_predict)
print('accuary rucy:', acc)

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis =1)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
y_submit += 3
submission['quality'] = y_submit
#submission.to_csv(path + 'sample_submission100.csv')
submission.to_csv(path + 'wine_' + date + '.csv')

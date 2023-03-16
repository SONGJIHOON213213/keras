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
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
path = './_data/telephone/'  
train_csv = pd.read_csv(path + 'train.csv', 
                        index_col=0) 

print(train_csv)
print(train_csv.shape) #출력결과 (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0) 
              
print(test_csv)       
print(test_csv.shape)  

train_csv = train_csv.dropna()

# 1.4 x, y 분리
x = train_csv.drop(['전화해지여부'], axis=1)
y = train_csv['전화해지여부']




# 1.5 train, test 분리 과적합방지
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1234, shuffle=True,stratify=y)

# 2. 스케일링
scaler = MinMaxScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv =scaler.transform(test_csv)


# # # # # 2. 모델구성
input1 = Input(shape=(12,))
dense1 = Dense(60)(input1)
dense2 = Dense(50, activation='relu')(dense1)
drop2 = Dropout(0.1)(dense2)
dense3 = Dense(40, activation='relu')(drop2)
drop3 = Dropout(0.1)(dense3)
dense4 = Dense(30, activation='relu')(drop3)
drop4 = Dropout(0.1)(dense4)
dense5 = Dense(20, activation='relu')(drop4)
drop5 = Dropout(0.1)(dense5)
output1 = Dense(1, activation='sigmoid')(drop5)
model = Model(inputs=input1, outputs=output1)

class_weight = {0:3000,1 : 26000}

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True,patience=80) # 기도 메타

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x_train, y_train, epochs=2000, batch_size=60, validation_split=0.2, callbacks=[es], class_weight=class_weight)


# model.save('./_save/keras26_3_save_model.h5')
# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result:', result)

y_predict = np.round(model.predict(x_test)).astype(int)

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
print('acc:', acc)
print('f1 score:', f1)

y_submit = np.round(model.predict(test_csv)).astype(int)#.flatten()
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['전화해지여부'] = y_submit
submission.to_csv(path + 'telephone_submission.csv')

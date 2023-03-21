from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np


# 1. 데이터
# 1.1 경로, 가져오기
path = './keras/_data/houseprice/'


train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항
print(train_csv.shape, test_csv.shape)
print(train_csv.columns, test_csv.columns) 

# 1.3 결측지
print(train_csv.isnull().sum())

# 1.4 라벨인코딩( object 에서 )
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
print(len(train_csv.columns))
print(train_csv.info())
train_csv=train_csv.dropna()

x = train_csv.drop(['SalePrice'], axis = 1)
y = train_csv['SalePrice']

print(x.shape)
print(y.shape)
print(test_csv.shape)
print(train_csv.shape)
x = np.array(x)
x = x.reshape(1121,79,1,1)#1459

test_csv = np.array(test_csv)
test_csv = test_csv.reshape(1459,79,1,1)#1121,

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=555,shuffle=True)

# scaler = MinMaxScaler() 
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)
# test_csv = scaler.fit_transform(test_csv)

#2.모델구성
model = Sequential()
model.add(Conv2D(64,(3,1),padding = 'same', input_shape = (79,1,1)))
model.add(Conv2D(10, 2, padding='same'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1))
model.summary() 

#3.컴파일 훈련
model.compile(loss='mse',optimizer ='adam')
es = EarlyStopping(monitor='val_loss', patience = 100,verbose = 1, mode='min',restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10, batch_size=30, verbose=1, validation_split=0.2, callbacks=[es])
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

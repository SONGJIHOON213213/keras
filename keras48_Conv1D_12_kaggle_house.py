from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten,SimpleRNN,LSTM,Conv1D,Flatten
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import r2_score

# 1. 데이터
# 1.1 경로, 가져오기
path = './keras/_data/houseprice/'
path_save = './_save/houseprice/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
for i in train_csv.columns:
    if not i in test_csv.columns:
        print(i)

# 1.2 확인사항
print(train_csv.shape, test_csv.shape)
print(train_csv.columns, test_csv.columns)

# 1.3 결측지 확인
# print(train_csv.isnull().sum())

# 1.4 라벨인코딩( 으로 object 결측지 제거 )
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
# print(len(train_csv.columns))
# print(train_csv.info())
train_csv=train_csv.dropna()
# print(train_csv.shape)

# # 1.5 x, y 분리
x = train_csv.drop(['SalePrice'], axis=1)
y = train_csv['SalePrice']


# 1.6 train, test 분리

x = np.array(x)

test_csv = np.array(test_csv)
print(x.shape)
print(y.shape)
# 1.6 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = np.reshape(x_train,[-1,79,1])
x_test = np.reshape(x_test,[-1,79,1])

test_csv = np.reshape(test_csv,[-1,79,1])

# # # # #2. 모델구성
model = Sequential()
model.add(Conv1D(10,2,activation='relu', input_shape = (79,1)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(3))
# #컴파일 훈련
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size = 150,validation_split=0.2)

# 4. 평가 예측
loss = model.evaluate(x_test, y_test, verbose= 0)
print("loss : ", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score:',  r2)
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/wine/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
###라벨인코딩###
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
le = LabelEncoder()
aaa = le.fit(train_csv['type'])
print(aaa)
print(type(aaa))
print(aaa.shape)
print(np.unique(aaa, return_counts=True))

train_csv['type'] = aaa
print(train_csv)
train_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red','white']))
print(le.transform(['white','red']))
###라벨인코딩###

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))

# 1.3 결측지 제거
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())

# 1.4 x, y 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
#한개의 컬럼 quality

# 1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state = 444 , shuffle=True)

scaler = MinMaxScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv =scaler.transform(test_csv)

# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=9, 
          verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

def RMSE(y_test,y_predict): #RMSE 는 만든 변수명
    mean_squared_error(y_test,y_predict) #함수반환 = 리턴을사용
    return np.sqrt(mean_squared_error(y_test, y_predict)) #rmse 에 루트 씌워진다 np.sqrt 사용하면
rmse = RMSE(y_test,y_predict) #RMSE 함수사용
print("RMSE: ", rmse)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission.to_csv(path + 'sample_submission41.csv')

y_submit = model.predict(test_csv)
submission['quality'] = y_submit


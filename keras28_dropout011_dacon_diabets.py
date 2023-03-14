from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import RobustScaler, MaxAbsScaler 
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# 1.데이터
# 1.1 경로, 가져오기
path = './_data/dacon_diabets/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv  = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape) #(652, 9) (116, 8)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))

# 1.3 결측지 제거
train_csv = train_csv.dropna()

# 1.4 x, y 분리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']


# # 1.5 train, test 분리 과적합방지
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1234, shuffle=True)

# scaler = MinMaxScaler() # 0.0 711.0 #정규화란, 모든 값을 0~1 사이의 값으로 바꾸는 것이다
scaler = StandardScaler() #정답은 (49-50) / 1 = -1이다. 여기서 표준편차란 평균으로부터 얼마나 떨어져있는지를 구한 것이다. 
# scaler = MaxAbsScaler #최대절대값과 0이 각각 1, 0이 되도록 스케일링
# scaler = RobustScaler #중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
print(np.min(x), np.max(x))



# # 2. 모델구성
model = Sequential()
dense1 = model.add(Dense(64,input_shape=(8,)))
dense1 = model.add(Dropout(0.1))
dense2 = model.add(Dense(30, activation='relu'))
dense2 = model.add(Dropout(0.1))
dense3 = model.add(Dense(30, activation='relu'))
dense3 = model.add(Dropout(0.1))
dense4 = model.add(Dense(20, activation='relu'))
dense5 = model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=80)

model.fit(x_train, y_train, epochs=1200, batch_size=20, validation_split=0.2, verbose=1, callbacks=[es])

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = np.round(model.predict(x_test))

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc:', acc)


y_submit = np.round(model.predict(test_csv))
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['Outcome'] = y_submit
submission.to_csv(path + 'sample_submission80.csv')
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)    
print('r2 : ', r2)

def RMSE(y_test,y_predict): #RMSE 는 만든 변수명
    mean_squared_error(y_test,y_predict) #함수반환 = 리턴을사용
    return np.sqrt(mean_squared_error(y_test, y_predict)) #rmse 에 루트 씌워진다 np.sqrt 사용하면
rmse = RMSE(y_test,y_predict) #RMSE 함수사용
print("RMSE: ", rmse)

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
path_save = './_save/houseprice/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

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
print(len(train_csv.columns))
print(train_csv.info())
train_csv=train_csv.dropna()
print(train_csv.shape)

# 1.5 x, y 분리
x = train_csv.drop(['SalePrice'], axis=1)
y = train_csv['SalePrice']

print(x.shape)
x = np.array(x)
x = x.reshape(-1, 79, 1, 1) #디폴트로 나머지 다넣어준다

test_csv = np.array(test_csv)
test_csv = test_csv.reshape(-1, 79, 1, 1)
# 1.6 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

# 1.7 Scaler
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,1), padding='same', input_shape=(79, 1, 1)))
model.add(Conv2D(10, 2, padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10, batch_size=30, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

y_submit = model.predict(test_csv)
import numpy as np
import pandas as pd
y_submit = pd.DataFrame(y_submit)
y_submit = y_submit.fillna(0)
y_submit = np.array(y_submit)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['SalePrice'] = y_submit

#RNN
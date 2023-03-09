import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

# 1.데이터
# 1.1 경로, 가져오기
path = './_data/dacon_diabets/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

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

# # 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=80)
hist = model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])

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
submission.to_csv(path + 'sample_submission32325.csv')
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
# plt.title('당뇨병')
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# plt.grid()   #격자
# plt.legend() #범례
# plt.show()

# #acc: 0.7602040816326531
#주가예측 시험
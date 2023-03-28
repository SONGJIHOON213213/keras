import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))

def split_x(dt, st):
    a = []
    for i in range(len(dt)-st):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

# 1. 데이터
# 1.1 경로, 가져오기
path = './keras/_data/시험/'
path_save = './_save/samsung/'

datasets_samsung = pd.read_csv(path + '삼성전자 주가2.csv', index_col=0, encoding='cp949')
datasets_hyundai = pd.read_csv(path + '현대자동차.csv', index_col=0, encoding='cp949')

print(datasets_samsung.shape, datasets_hyundai.shape)
print(datasets_samsung.columns, datasets_hyundai.columns)
print(datasets_samsung.info(), datasets_hyundai.info())
print(datasets_samsung.describe(), datasets_hyundai.describe())
print(type(datasets_samsung), type(datasets_hyundai))

samsung_x = np.array(datasets_samsung.drop(['전일비', '종가'], axis=1))
samsung_y = np.array(datasets_samsung['종가'])
hyundai_x = np.array(datasets_hyundai.drop(['전일비', '종가'], axis=1))
hyundai_y = np.array(datasets_hyundai['종가'])

samsung_x = samsung_x[:180, :]
samsung_y = samsung_y[:180]
hyundai_x = hyundai_x[:180, :]
hyundai_y = hyundai_y[:180]

samsung_x = np.flip(samsung_x, axis=1)
samsung_y = np.flip(samsung_y)
hyundai_x = np.flip(hyundai_x, axis=1)
hyundai_y = np.flip(hyundai_y)

print(samsung_x.shape, samsung_y.shape)
print(hyundai_x.shape, hyundai_y.shape)

samsung_x = np.char.replace(samsung_x.astype(str), ',', '').astype(np.float64)
samsung_y = np.char.replace(samsung_y.astype(str), ',', '').astype(np.float64)
hyundai_x = np.char.replace(hyundai_x.astype(str), ',', '').astype(np.float64)
hyundai_y = np.char.replace(hyundai_y.astype(str), ',', '').astype(np.float64)

_, samsung_x_test, _, samsung_y_test, _, hyundai_x_test, _, hyundai_y_test = train_test_split(samsung_x, samsung_y, hyundai_x, hyundai_y, train_size=0.7, shuffle=False)
(samsung_x_train,samsung_y_train,hyundai_x_train,hyundai_y_train)=(samsung_x, samsung_y, hyundai_x, hyundai_y)

scaler = MinMaxScaler()
samsung_x_train = scaler.fit_transform(samsung_x_train)
samsung_x_test= scaler.transform(samsung_x_test)
hyundai_x_train = scaler.transform(hyundai_x_train)
hyundai_x_test = scaler.transform(hyundai_x_test)

timesteps = 20

samsung_x_train_split = split_x(samsung_x_train, timesteps)
samsung_x_test_split = split_x(samsung_x_test, timesteps)
hyundai_x_train_split = split_x(hyundai_x_train, timesteps)
hyundai_x_test_split = split_x(hyundai_x_test, timesteps)

samsung_y_train_split = samsung_y_train[timesteps:]
samsung_y_test_split = samsung_y_test[timesteps:]
hyundai_y_train_split = hyundai_y_train[timesteps:]
hyundai_y_test_split = hyundai_y_test[timesteps:]

print(samsung_x_train_split.shape)
print(hyundai_x_train_split.shape)

model = load_model('./_save/samsung/keras53_samsung2_sjh.h5')

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# 4. 평가, 예측
loss = model.evaluate([samsung_x_test_split, hyundai_x_test_split], [samsung_y_test_split, hyundai_y_test_split])
print('loss : ', loss)

samsung_x_predict = samsung_x_test[-timesteps:]
samsung_x_predict = samsung_x_predict.reshape(1, timesteps, 14)
hyundai_x_predict = hyundai_x_test[-timesteps:]
hyundai_x_predict = hyundai_x_predict.reshape(1, timesteps, 14)

predict_result = model.predict([samsung_x_predict, hyundai_x_predict])

print("내일의 종가는 바로바로 : ", np.round(predict_result[0], 2))
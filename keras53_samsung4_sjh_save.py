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
    for i in range(len(dt)-st-1):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/시험/'
path_save = './_save/samsung/'

datasets_hyundai  = pd.read_csv(path + 'pollution.csv', index_col=0, encoding='cp949')
datasets_samsung = pd.read_csv(path + '삼성전자 주가3.csv', index_col=0, encoding='cp949')
print(datasets_samsung.columns)
print(datasets_hyundai.columns)

samsung_x = np.array(datasets_samsung.drop(['전일비', '금액(백만)','신용비','외인비'], axis=1))
samsung_y = np.array(datasets_samsung['시가'])
hyundai_x = np.array(datasets_hyundai.drop(['전일비', '금액(백만)','신용비','외인비'], axis=1))
hyundai_y = np.array(datasets_hyundai['시가'])


samsung_x = samsung_x[:200, :]
samsung_y = samsung_y[:200]
hyundai_x = hyundai_x[:200 :]
hyundai_y = hyundai_y[:200]

samsung_x = np.flip(samsung_x, axis=1)
samsung_y = np.flip(samsung_y)
hyundai_x = np.flip(hyundai_x, axis=1)
hyundai_y = np.flip(hyundai_y)

samsung_x = np.char.replace(samsung_x.astype(str), ',','').astype(np.float64)
samsung_y = np.char.replace(samsung_y.astype(str), ',','').astype(np.float64)
hyundai_x = np.char.replace(hyundai_x.astype(str), ',','').astype(np.float64)
hyundai_y = np.char.replace(hyundai_y.astype(str), ',','').astype(np.float64)


_, samsung_x_test, _, samsung_y_test, _, hyundai_x_test, _, hyundai_y_test = train_test_split(samsung_x, samsung_y, hyundai_x, hyundai_y, train_size=0.7, shuffle=False)
(samsung_x_train,samsung_y_train,hyundai_x_train,hyundai_y_train)=(samsung_x, samsung_y, hyundai_x, hyundai_y)

def scale(x_train,x_test):
    scaler = MinMaxScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    return x_train,x_test
samsung_x_train, samsung_x_test=scale(samsung_x_train, samsung_x_test)
hyundai_x_train, hyundai_x_test=scale(hyundai_x_train, hyundai_x_test)

timesteps = 20

samsung_x_train_split, samsung_x_test_split, hyundai_x_train_split,hyundai_x_test_split = map(lambda x: split_x(x, timesteps), [samsung_x_train, samsung_x_test, hyundai_x_train, hyundai_x_test])

samsung_y_train_split, samsung_y_test_split, hyundai_y_train_split,hyundai_y_test_split = map(lambda y: y[(timesteps+1):], [samsung_y_train, samsung_y_test, hyundai_y_train, hyundai_y_test])

# # 2. 모델구성

input1=Input(shape=(samsung_x_train_split.shape[1:]))
input2=Input(shape=(hyundai_x_train_split.shape[1:]))
merge=Concatenate()((input1,input2))
layer=SimpleRNN(32)(merge)
layer=Dense(16,activation='linear')(layer)
layer=Dense(16,activation='linear')(layer)
layer=Dense(16,activation='relu')(layer)
layer=Dense(16,activation='linear')(layer)
layer=Dense(16,activation='linear')(layer)
output1=Dense(1)(layer)
output=(output1)
model=Model(inputs=(input1,input2),outputs=output)
model.summary()


# # 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
hist = model.fit([samsung_x_train_split, hyundai_x_train_split], [ hyundai_y_train_split], epochs=2000, batch_size=4, validation_split=0.2, callbacks=[es])

model.save(path_save + 'keras53_hyundai16_sjh.h5')

# # 4. 평가, 예측
loss = model.evaluate([samsung_x_test_split, hyundai_x_test_split], [ hyundai_y_test_split])
print('loss : ', loss)

samsung_x_predict = samsung_x_test[-timesteps:]
samsung_x_predict = samsung_x_predict.reshape([1]+list(samsung_x_train_split.shape[1:]))
hyundai_x_predict = hyundai_x_test[-timesteps:]
hyundai_x_predict = hyundai_x_predict.reshape([1]+list(hyundai_x_train_split.shape[1:]))

predict_result = model.predict([samsung_x_predict, hyundai_x_predict])


print("이틀뒤의 현대의 시가 : ", np.round(predict_result,2))

plt.plot(range(len(hyundai_y_train_split)),hyundai_y_train_split,label='origin')
plt.plot(range(len(hyundai_y_train_split)),model.predict([samsung_x_train_split, hyundai_x_train_split]),label='model')
plt.legend()
plt.show()

#keras53_hyundai18_sjh.h5
#[[179665.]]
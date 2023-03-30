# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기
# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라

# 제공된 데이터 외 추가 데이터 사용 금지

# 1. 삼성전자 28일(화) 종가 맞추기(점수배점 0.3)
# 2. 삼성전자 29일(수) 아침 시가 맞추기(점수배점 0.7)
# 메일 제목 : 송지훈 [삼성 1차] 60,350.07원
# 첨부 파일 : keras53_samsung2_jsw_submit.py
# 첨부 파일 : keras53_samsung4_jsw_submit.py
# 가중치    : _save/samsung/keras53_samsung2_jsw.j5
# 가중치    : _save/samsung/keras53_samsung4_jsw.j5
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input,Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import datetime
import random
import tensorflow as tf

seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
path = './keras/_data/시험/'
path_save = './_save/samsung/'



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
path = './_data/시험/'
path_save = './_save/samsung/'

datasets_samsung = pd.read_csv(path + '삼성전자 주가3.csv', index_col=0, encoding='cp949')
datasets_hyundai = pd.read_csv(path + '현대자동차2.csv', index_col=0, encoding='cp949')

# print(datasets_samsung.shape, datasets_hyundai.shape)
# print(datasets_samsung.columns, datasets_hyundai.columns)
# print(datasets_samsung.info(), datasets_hyundai.info())
# print(datasets_samsung.describe(), datasets_hyundai.describe())
# print(type(datasets_samsung), type(datasets_hyundai))

samsung_x = np.array(datasets_samsung.drop(['전일비', '종가'], axis=1))
samsung_y = np.array(datasets_samsung['종가'])
hyundai_x = np.array(datasets_hyundai.drop(['전일비', '종가'], axis=1))
hyundai_y = np.array(datasets_hyundai['종가'])


samsung_x = samsung_x[:230, :]
samsung_y = samsung_y[:230]
hyundai_x = hyundai_x[:230, :]
hyundai_y = hyundai_y[:230]

samsung_x, samsung_y = np.flip(samsung_x, axis=1), np.flip(samsung_y)
hyundai_x, hyundai_y = np.flip(hyundai_x, axis=1), np.flip(hyundai_y)

# print(samsung_x.shape, samsung_y.shape)
# print(hyundai_x.shape, hyundai_y.shape)


samsung_x, samsung_y, hyundai_x, hyundai_y = map(lambda x: np.char.replace(x.astype(str), ',', '').astype(np.float64), [samsung_x, samsung_y, hyundai_x, hyundai_y])

_, samsung_x_test, _, samsung_y_test, _, hyundai_x_test, _, hyundai_y_test = train_test_split(samsung_x, samsung_y, hyundai_x, hyundai_y, train_size=0.7, shuffle=False)
(samsung_x_train,samsung_y_train,hyundai_x_train,hyundai_y_train)=(samsung_x, samsung_y, hyundai_x, hyundai_y)



scaler = MinMaxScaler()
samsung_x_train, samsung_x_test, hyundai_x_train, hyundai_x_test = map(scaler.fit_transform, [samsung_x_train, samsung_x_test, hyundai_x_train, hyundai_x_test])

# timesteps = 20
timesteps = 20
samsung_x_train_split, samsung_x_test_split, hyundai_x_train_split ,hyundai_x_test_split = map(lambda x: split_x(x, timesteps), [samsung_x_train, samsung_x_test, hyundai_x_train, hyundai_x_test])

samsung_y_train_split, samsung_y_test_split, hyundai_y_train_split, hyundai_y_test_split = map(lambda y: y[timesteps:], [samsung_y_train, samsung_y_test, hyundai_y_train, hyundai_y_test])
                                                                                                                                       



# 2. 모델구성
# 2.1 모델1
input1 = Input(shape=(timesteps, 14))
dense1 = LSTM(40, activation='relu', name='sam1')(input1)
dense2 = Dense(200, activation='relu', name='sam2')(dense1)
dense3 = Dense(300, activation='relu', name='sam3')(dense2)
output1 = Dense(110, activation='relu', name='sam4')(dense3)

# 2.2 모델2
input2 = Input(shape=(timesteps, 14))
dense11 = LSTM(30, name='huyn1')(input2)
dense12 = Dense(90, name='huyn2')(dense11)
dense13 = Dense(100, name='huyn3')(dense12)
dense14 = Dense(90, name='huyn4')(dense13)
output2 = Dense(70, name='output2')(dense14)

# 2.3  Concatenate
nerge1 = Concatenate(name='mg1')([output1, output2])
reshape1=Reshape((90,2))(nerge1)
dense1 = LSTM(40, activation='relu', name='sam5')(reshape1)
dense2 = Dense(200, activation='relu', name='sam6')(dense1)
dense3 = Dense(300, activation='relu', name='sam7')(dense2)
output1 = Dense(110, activation='relu', name='sam8')(dense3)
# 2.4 분기1

nerge2 = Dense(80, activation='relu', name='ng2')(output1)
nerge3 = Dense(50, activation='relu', name='ng3')(nerge2)
hidden_output = Dense(50, name='last')(nerge3)

# 2.6 모델 조립
model = Model(inputs=[input1, input2], outputs=[output1])

model.summary()
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
hist = model.fit([samsung_x_train_split, hyundai_x_train_split], [samsung_y_train_split], epochs=2000, batch_size=50,
                 validation_data=([samsung_x_test_split, hyundai_x_test_split], [samsung_y_test_split, hyundai_y_test_split]), callbacks=[es])

model.save(path_save + 'keras53_samsung2_sjh.h5')


# 4. 평가, 예측

loss = model.evaluate([samsung_x_test_split, hyundai_x_test_split], [samsung_y_test_split])
print('loss : ', loss)

for_r2=model.predict([samsung_x_test_split, hyundai_x_test_split])
print(samsung_y_test_split.shape,for_r2.shape)
r2_samsung=r2_score(samsung_y_test_split,for_r2)
# r2_hyundai=r2_score(hyundai_y_test_split,for_r2[1])

print(f'결정 계수 : {r2_samsung}')

samsung_x_predict = np.reshape(samsung_x_test[-timesteps:], (1, timesteps, 14))
hyundai_x_predict = np.reshape(hyundai_x_test[-timesteps:], (1, timesteps, 14))

predict_result = model.predict([samsung_x_predict, hyundai_x_predict])

print("내일의 종가 : ", np.round(predict_result, 2)) 

def val_split(x):
    return x[2*len(x)//5:]
x1_val=val_split(samsung_x_train_split)
x2_val=val_split(hyundai_x_train_split)
y1_val=val_split(samsung_y_train_split)
y2_val=val_split(hyundai_y_train_split)


import matplotlib.pyplot as plt
y_pred = model.predict([x1_val,x2_val])
plt.subplot(1,2,1)
plt.plot(range(len(y1_val)),y1_val,label='real')
plt.plot(range(len(y1_val)),y_pred,label='model')
plt.legend
# plt.subplot(1,2,2)
# plt.plot(range(len(y2_val)),y2_val,label='real')
# plt.plot(range(len(y2_val)),y_pred[1],label='model')
# plt.legend()
plt.show()



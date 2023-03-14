# save_model과 비교 저장할때 지표값,훈련시간 파일에 넣어줘
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import RobustScaler, MaxAbsScaler 
from tensorflow.python.keras.callbacks import EarlyStopping


#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets['target']
print(x.shape,y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
     shuffle= True, random_state=942, 
     train_size= 0.8)# random_state=123) #내가 모르면 2를쓰자 데이터가 한쪽으로 치우칠수 있으므로 y라벨값 비율의개수만큼 빼준다.
# scaler = MinMaxScaler() # 0.0 711.0 #정규화란, 모든 값을 0~1 사이의 값으로 바꾸는 것이다
scaler = StandardScaler() #정답은 (49-50) / 1 = -1이다. 여기서 표준편차란 평균으로부터 얼마나 떨어져있는지를 구한 것이다. 
# scaler = MaxAbsScaler #최대절대값과 0이 각각 1, 0이 되도록 스케일링
# scaler = RobustScaler #중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
##########################한줄로 정리가능 42,43을 45로
# x = scaler.fit(x_train)
# x_train = scaler.transform(x_train)

x_train = scaler.fit_transform(x_train)
############################
x_test = scaler.transform(x_test) 
print(np.min(x), np.max(x))


#2.모델

input1 = Input(shape=(13,)) #input-> desen1 ->dense 2->desne3 -> output1-> 모델순서
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)

#model.save('./_save/keras26_1_save_model.h5')#  .은현재위치 /통상적으로 확장자는h5

#데이터가 3차원이면
#(1000,100, 1) ->>> input_shape=(100,1)
#데이터가 4차원이면
#(60000,32,32, 3) ->>> input_shape = (32,32,3)
6

#3.컴파일,훈련
model.compile(loss = 'mse',optimizer = 'adam') #MSE 종류면 최소값  

import datetime 
date = datetime.datetime.now()
print(date)
date = date.strftime("%m%d_%H%M")
print(date) # 0314_1115


filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdfs'




from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience= 10, mode= 'min', verbose=1, #restore_best_weights = True 
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1 ,
                      save_best_only=True, filepath="".join([filepath,'k27_',date,'_',filename])
)
model.fit(x_train,y_train, epochs=1000,batch_size =32 , callbacks=[es, mcp],validation_split=0.2)#호출하겟다

# 4. 평가, 예측
from sklearn.metrics import r2_score

print('==========================기본출력============================')
loss = model.evaluate(x_test, y_test, verbose= 0)
print("loss : ", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score:',  r2)



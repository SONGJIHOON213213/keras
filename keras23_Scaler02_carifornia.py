import numpy as np
from sklearn.datasets  import fetch_california_housing
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
datasets = fetch_california_housing()
# print(datasets.DESCR) #판다스 describe()
# print(datasets.feature_names)  # 판다스 columes() #Feature ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape,y.shape) # (150, 4) (150,)
# print(x)
# print(y) #데이터가많아지면,다중분류면 y의 라벨 확인 #컬럼-> 디멘션-> 노드 갯수

print(x.shape,y.shape)
print(x)
print(y)
print('y의 라벨값:', np.unique(y))
###############################요지점에서 원핫을 해야겠죠?

from keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) 
## y를(150,)-> (150, 3) 판다스.겟더미 # 사이킷에 원핫인코더
###################################################
x_train, x_test, y_train, y_test = train_test_split(x, y, 
     shuffle= True, random_state=942, 
     train_size= 0.8)# random_state=123) #내가 모르면 2를쓰자 데이터가 한쪽으로 치우칠수 있으므로 y라벨값 비율의개수만큼 빼준다.
# scaler = MinMaxScaler() # 0.0 711.0 #정규화란, 모든 값을 0~1 사이의 값으로 바꾸는 것이다
scaler = StandardScaler() #정답은 (49-50) / 1 = -1이다. 여기서 표준편차란 평균으로부터 얼마나 떨어져있는지를 구한 것이다. 
# scaler = MaxAbsScaler #최대절대값과 0이 각각 1, 0이 되도록 스케일링
# scaler = RobustScaler #중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
print(np.min(x), np.max(x))


#2.모델

# model = Sequential()
# model.add(Dense(30,input_shape=(13,)))#input_dim =13 이면 13,로쓰면된다
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape=(8,)) #input-> desen1 ->dense 2->desne3 -> output1-> 모델순서
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(6)(dense3)
model = Model(inputs = input1, outputs = output1)

#데이터가 3차원이면
#(1000,100, 1) ->>> input_shape=(100,1)
#데이터가 4차원이면
#(60000,32,32, 3) ->>> input_shape = (32,32,3)

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=10)

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss:',loss)

#일요일짜과제 StandardScaler내용적기
#함수<--재사용 함수형모댈
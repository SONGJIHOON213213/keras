#input_dim 은  3차원 ~ 4차원부터 할 수 없다.
#이미지는 4차원으로 들어간다.
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))
print(x)

# 0.0 1.0

x_train, x_test, y_train, y_test =  train_test_split(
    x,y, train_size=0.8, random_state=333
)

# scaler = MinMaxScaler() # 0.0 711.0 #정규화란, 모든 값을 0~1 사이의 값으로 바꾸는 것이다
# scaler = StandardScaler() #정답은 (49-50) / 1 = -1이다. 여기서 표준편차란 평균으로부터 얼마나 떨어져있는지를 구한 것이다. 
# scaler = MaxAbsScaler #최대절대값과 0이 각각 1, 0이 되도록 스케일링
scaler = RobustScaler #중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
print(np.min(x), np.max(x))


#2.모델
model = Sequential()
model.add(Dense(1,input_shape=(13,)))#input_dim =13 이면 13,로쓰면된다


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
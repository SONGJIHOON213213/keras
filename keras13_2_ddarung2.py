#데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './_data/ddarung/'   #점 하나 현재폴더의밑에 점하나는 스터디 
path_save = './_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv' ,
                        index_col=0)
# train_csv = pd.read_csv('./_data/ddarung/train.csv')

print(train_csv)
print(train_csv.shape) # 출력결과 (1459,11)




#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32, 
          verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

submission.to_csv(path_save + 'submit_083434_15121.csv')


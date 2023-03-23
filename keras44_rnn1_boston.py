import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets['target']

print(x.shape, y.shape)  # (7, 3)(7,)
x = x.reshape(-1, 13, 1)  # [[1],[2],[3]],[[2],[3],[4], ........]
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=942, train_size=0.8)

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(600,activation='relu', input_shape = (13,1)))
model.add(Dense(400, activation ='relu'))
model.add(Dense(200, activation ='relu'))
model.add(Dense(100, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(1,  activation = 'linear'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10,batch_size=60 , verbose=1, validation_split=0.2)

# 4. 평가 예측

loss = model.evaluate(x_test, y_test, verbose= 0)
print("loss : ", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score:',  r2)
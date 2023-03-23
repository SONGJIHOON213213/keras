import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)
  # (7, 3)(7,)
x = x.reshape(-1, 10, 1)  # [[1],[2],[3]],[[2],[3],[4], ........]

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=942, train_size=0.8)
print(x.shape, y.shape)
# 2. 모델 구성
model = Sequential()
model.add(Conv1D(10,2,activation='relu', input_shape = (10,1)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100,batch_size=60 , verbose=1, validation_split=0.2)

# 4. 평가 예측

loss = model.evaluate(x_test, y_test, verbose= 0)
print("loss : ", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score:',  r2)
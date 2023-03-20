from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)  # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=942, train_size=0.8)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)

x_train= np.reshape(x_train,(x_train.shape[0],x_train.shape[1]//4,2,2))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]//4,2,2))
print(x_train.shape)

print(np.min(x), np.max(x))




# 2. 모델링
from tensorflow.keras.layers import Flatten
model = Sequential()
model.add(Conv2D(filters=18
                 ,input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
                 ,kernel_size=(2,2)
                 ,padding='valid'
                 ,activation='linear'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(1, activation='softmax'))


#3.컴파일,훈련
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size= 80, validation_split = 0.2)

# 4. 평가, 예측
from sklearn.metrics import r2_score

print('==========================기본출력============================')
loss = model.evaluate(x_test, y_test, verbose= 0)
print("loss : ", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score:',  r2)
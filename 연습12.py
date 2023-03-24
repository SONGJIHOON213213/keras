from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np

# 1. 데이터
path = 'keras/_data/jena_climate/'

dataset = np.loadtxt(path + 'jena_climate_2009_2016.csv', delimiter=',', skiprows=1)
print(dataset.shape) #(420551, 14)

def split_x(dataset, timesteps):
    forlistgen = (dataset[i:i+timesteps] for i in range(len(dataset)-timesteps+1))
    return np.array(list(forlistgen))

timesteps = 5
dataset = split_x(dataset, timesteps)

x = dataset[:, :-1]
y = dataset[:, -1, 3]

datasetforpred = np.array(range(106))
x_test = split_x(datasetforpred, timesteps)
y_test = x_test[:, -1] + 1

#2.모델
model = Sequential()
model.add(LSTM(128, kernel_regularizer='l2', input_shape=(timesteps, x.shape[2])))
model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(Dense(32, kernel_regularizer='l2'))
model.add(Dense(1))

#3.컴파일
model.compile(optimizer = 'adam',loss="mse", metrics=["mae"])
model.ftx(x,y,epochs = 10,verbose = 1 , validation_split = 0.2,batch_size = 60)
# #4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:',loss)
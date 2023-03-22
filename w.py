import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

dataset = np.array(range(1, 101))
timesteps = 5
x_predict = np.array(range(96, 106))

def split_x(dataset, timesteps):
    forlistgen = (dataset[i:i+timesteps] for i in range(len(dataset)-timesteps+1))
    return np.array(list(forlistgen))

bbb = split_x(dataset, timesteps)
x = bbb[:, :-1]
y = bbb[:, -1]

y_predict = np.array(range(106))

model = Sequential()
model.add(LSTM(units=32, input_shape=(timesteps, 5), activation='linear'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x.reshape(-1, timesteps, 5), y, epochs=100, verbose=0)

y_pred = model.predict(x_predict.reshape(1, timesteps, 1))
print(y_pred)
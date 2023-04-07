import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Load data
path = 'C:/study/_data/시험/'
datasets1 = pd.read_csv(path + 'train.csv', index_col=0, encoding='cp949')
datasets2 = pd.read_csv(path + 'test.csv', index_col=0)

# data1.info()

# Make the index unique


# Lowercase and replace spaces in column names
datasets1.columns = datasets1.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(','').str.replace(')','')
datasets2.columns = datasets2.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(','').str.replace(')','')

# Prepare data for LSTM model
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform
x_train = []
y_train = []



scaled_data = scaler.fit_transform(datasets1)
# Prepare training data for LSTM model
# for i in range(60, len(scaled_data)):
#     x_train.append(scaled_data[i-60:i, :])
#     y_train.append(scaled_data[i, 0])

# # Convert lists to arrays
# x_train = np.array(x_train)
# y_train = np.array(y_train)

# # Define LSTM model
# lstm_model = Sequential()
# lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
# lstm_model.add(Dropout(0.2))
# lstm_model.add(LSTM(units=50, return_sequences=True))
# lstm_model.add(Dropout(0.2))
# lstm_model.add(LSTM(units=50))
# lstm_model.add(Dropout(0.2))
# lstm_model.add(Dense(units=1))

# # Compile model
# lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# # Train model
# lstm_model.fit(x_train, y_train, epochs=50, batch_size=32)

# # Prepare test data for LSTM model
# x_test = scaled_data[-60:, :]
# x_test = np.array([x_test])

# # Make predictions
# predictions = []
# for i in range(30):
#     prediction = lstm_model.predict(x_test)
#     predictions.append(prediction)
#     x_test = np.append(x_test[:,1:,:], prediction.reshape(1,1,1), axis=1)

# # Convert predictions to original scale
# predictions = np.array(predictions).reshape(-1,1)
# predictions = scaler.inverse_transform(predictions)

# # Print predictions
# print(predictions)
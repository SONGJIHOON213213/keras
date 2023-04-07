import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load data
path = 'C:/study/_data/exam/'
datasets1 = pd.read_csv(path + 'train.csv', index_col=0, encoding='cp949')
datasets2 = pd.read_csv(path + 'test.csv', index_col=0)

# Make the index unique
datasets1.index = [f"{i}_{datasets1.index[i]}" for i in range(len(datasets1))]
datasets2.index = [f"{i+len(datasets1)}_{datasets2.index[i]}" for i in range(len(datasets2))]

# Merge datasets
datasets = pd.concat([datasets1, datasets2], axis=1)

# Define target variable and features
target_var = 'target'  # Replace 'target' with the name of your target variable
features = [col for col in datasets.columns if col != target_var]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(datasets[features], datasets[target_var], test_size=0.2, random_state=42)

# Preprocess data using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
y_pred = model.predict(X_test)
print('R2 score:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))

# Save model
model.save('my_model.h5')
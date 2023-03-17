from tensorflow.keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Reshape the data
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

    # Scale the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(60000, 28 * 28)).reshape(60000, 28, 28, 1)
x_test = scaler.transform(x_test.reshape(10000, 28 * 28)).reshape(10000, 28, 28, 1)

    # One-hot encode the target labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

    # Create the model
model = Sequential()
model.add(Conv2D(7, (15,15), padding='valid', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=4, kernel_size=(9,9), padding='same', activation='relu'))
model.add(Conv2D(10, (5,5)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))

    # Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # Train the model
es = EarlyStopping(monitor='val_acc', patience=15, mode='max', restore_best_weights=True, verbose=1)
hist = model.fit(x_train, y_train, epochs=5, batch_size=3000, verbose=1, validation_split=0.2, callbacks=[es])

    # Evaluate the model
results = model.evaluate(x_test, y_test)
print('Results:', results)

    # Make predictions
y_predict = np.argmax(model.predict(x_test), axis=-1)
y_test = np.argmax(y_test, axis=-1)

    # Calculate accuracy
acc = accuracy_score(y_test, y_predict)
print('Accuracy score:', acc)

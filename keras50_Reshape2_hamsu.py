from tensorflow.keras.datasets import  fashion_mnist
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Conv1D,Reshape,LSTM,Input,Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) =fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255.
print(x_train.shape,y_train.shape)


# 2. 모델 구성
input_layer = Input(shape=(28, 28, 1))
x = Conv2D(32, (3,3), activation='relu')(input_layer)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
# 3. 컴파일, 훈련
model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model.fit(x_train, y_train, epochs=60, batch_size=80, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측 
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy:", accuracy)
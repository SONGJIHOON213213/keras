from tensorflow.keras.layers import Conv1D,Flatten,Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import pandas as pd
#1.데이터
path = './keras/_data/dacon_diabets/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv  = pd.read_csv(path + 'test.csv', index_col=0)

# 1.3 결측지 제거
train_csv = train_csv.dropna()

# 1.4 x, y 분리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1234, shuffle=True)#stratify 골고루 섞다

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.max(x_train))

# # 0.0
# # 1.0
print(x_train.shape)
print(x_test.shape)
x_train = np.reshape(x_train,(456,8,1))
x_test = np.reshape(x_test,(196,8,1))
# (105, 4)
# (45, 4) 
print(x_train.shape)
print(x_test.shape) 

# # # # #2. 모델구성
model = Sequential()
model.add(Conv1D(10,2,activation='relu', input_shape = (8,1)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(2))

# # # #3.컴파일,훈련
model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics = 'acc')
model.fit(x_train,y_train,epochs = 5,verbose = 1 , validation_split = 0.2,batch_size = 60)

# #4.평가,예측
eva=model.evaluate(x_test,y_test) #eva를 쓰면 acc,loss,가 안에 들어간다.
print('accuracy :',eva[1])
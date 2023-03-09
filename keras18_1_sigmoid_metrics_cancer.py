import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#과제 리스크,딕셔너리,큐플 어떤거인지 공부 # 판다스 : .describe()
print(datasets.DESCR)                      # 판다스 : columns()
print(datasets.feature_names)

x = datasets['data']
y = datasets.target

print(x.shape, y.shape)#(569, 30) #열 30개 특성 30
# print(y)                      #(569,)    #아웃트1개 스칼라569, 벡터1개

x_train, x_test,y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

#2. 모델구성
model = Sequential()
model.add(Dense(10,input_dim=30,activation='relu'))
model.add(Dense(9, activation='linear'))                      
model.add(Dense(8, activation='linear'))
model.add(Dense(7, activation='linear'))                      
model.add(Dense(1, activation='sigmoid'))
#이진분류데이터는 문제는 마지막에 sigmoid로수정
#loss를 binary_crossentropy

#3. 컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam' , 
              metrics=['accuracy','mse',]#'acc','mse','mean_squared_error'] 
              )
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=100)
hist = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])

#4.평가,예측
result = model.evaluate(x_test,y_test)
print('relust:', result)
y_predict = np.round(model.predict(x_test))
# print("==============================")
# print(y_test[:5])
# print(y_predict[:5])
# print(np.round(y_predict[:5]))
# print("=============================")
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc:', acc)

#활성화함수중에 0과1로 한정시키는함수
# 이진분류 마지막 시그모이드 
# mse 로스를 쓰게되면 실수값이 나오기때문에 bycrossentroy로 사용한다
# accuracy 를 넣게되면 훈련과정에 loss값 acc값이나옴
# 2개이상은 리스트 list[] 딕셔너리 키벨류 얼리스탑핑[es]
# relust 결과값 [loss,accracy,mse값]

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('유방암')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.grid()   #격자
plt.legend() #범례
plt.show()
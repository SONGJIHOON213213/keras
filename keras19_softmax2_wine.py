import numpy as np
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf


#1. 데이터
datasets = load_wine()
print(datasets.DESCR) #판다스 describe()
print(datasets.feature_names)  # 판다스 columes() #Feature ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape,y.shape) # (150, 4) (150,)
print(x)
# # print(y) #데이터가많아지면,다중분류면 y의 라벨 확인 #컬럼-> 디멘션-> 노드 갯수

print(x.shape,y.shape)
print(x)
print(y)
print('y의 라벨값:', np.unique(y))
print(y)
# # ###############################요지점에서 원핫을 해야겠죠?

# from keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) 
# ## y를(150,)-> (150, 3) 판다스.겟더미 # 사이킷에 원핫인코더
# ###################################################
# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#      shuffle= True, random_state=123, 
#      train_size= 0.8, 
#      stratify=y)# random_state=123) #내가 모르면 2를쓰자 데이터가 한쪽으로 치우칠수 있으므로 y라벨값 비율의개수만큼 빼준다.
# print(np.unique(y_train,return_counts=True))

# # # #튜닝자동화

# # # # #2. 모델구성
# model = Sequential()
# model.add(Dense(50, input_dim=13, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(3, activation='softmax')) #3개노드를 뽑기위해서


# # # # #3 컴파일, 훈련
# model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=80)
# hist = model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])

# # #accuracy_score 를 사용해서 스코어를 빼세요
# # # # 4. 평가, 예측


# # # # # 4. 평가, 예측
# result = model.evaluate(x_test, y_test)
# print(result)
# print('loss',result[0])
# print('acc', result[1])

# y_predict = model.predict(x_test)
# print(y_predict.shape)
# y_predict = np.argmax(y_predict, axis=-1)
# print(y_predict.shape)
# y_true = np.argmax(y_test, axis=-1)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_true, y_predict)
# print('accuracy-_score : ', acc)
# acc: 0.9166666666666666
#아그맥스 위치값을 빼준다
# ex[123] 제일 높은 위치값 반환
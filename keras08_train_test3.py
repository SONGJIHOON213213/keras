import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    #train_size=0.7, # 70프로사이즈
    test_size=0.3,
    random_state=1234, #<-랜덤스테이트 고정안하면 값이 게속 바뀐다 랜덤값을 고정하는역활
    shuffle=True
)
print(x_train)
print(x_test)

#랜덤값을 1부터10까지 뽑아내지만 파라미터 튜닝을하는데 데이터 값이 바뀐다면 훈련할떄 정확한 값을 알기 어렵다
#그래서 랜덤시드가 이것들을 잡아줘서 정확한 값을측정가능하게한다. 
#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=1)
 
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :' ,loss)
result = model.predict([[11]])
print('[11]의 예측값 :', result)


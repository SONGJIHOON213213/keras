import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 행무시 열우선 #2가지의 특성을 가지고 하나의 결과를 찾기 #만약에 10행3열이면 dim = 3 
x = np.array(
    [[1, 1], #10행 2열 열(컬럼,특성,피처) 행무시 열우선
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 1],
    [7, 1.4],
    [8, 1.4],
    [9, 1.6],
    [10, 1.4]]
) 
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) #개체 하나가 스칼라 스칼라가 모이면 벡터 벡터가 모이면 행렬 노드 처음에 2개 -> 3개 -> 5개 -> 1개

print(x.shape) #열 피쳐 컬럼 동일이야기 #(10,2)
print(y.shape)#(10,)

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=2))#열이 2개라 딤은 2개 인풋디멘션 차원 열의갯수와 동일 그래서 =2
model.add(Dense(5))
model.add(Dense(4))  #실질적 훈련 60번
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #모델.컴파일 오차 <-
model.fit(x, y, epochs=30, batch_size=5) #훈련 2번씩

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ' ,loss)

result = model.predict([[10, 1.4]])
print('[[10 1.4]]의 예측값 :', result)

#5/5 [==============================] - 배치사이즈가 5면 0s 997us/step - loss: 246.0495 5번에 나눠서 훈련시켰다. 
# 모델평가는 웨이트로
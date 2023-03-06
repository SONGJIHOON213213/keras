
sigmoid
#시그모이드 함수는 0->1까지 서서히 변하는 함수입니다.
# 이런 서서히 변화하는 비선형적인 특성으로 인해 신경망 활성화 함수로 활용됩니다
model.add(Dense(20, input_dim=10, activation = 'sigmoid'))

#R2 r2 결정계수
# from sklearn.metrics import r2_score #predict 예측한 y값
# r2 = r2_score(y_test, y_predict)

from tensorflow.keras.layers import Dense, LeakyReLU

# 결정계수는 회귀 분석에 의해 도출된 목적 변수의 예측 값이, 
실제 목적 변수의 값과 어느정도 일지하는가를 표시하는 지표이다. 
회귀분석에는 y=ax+b이라는 식으로 표시할 수 있는 단일 
회귀 분석과 설명변수가 여러 개 있는 다중 회귀 분석이 있다.

Relu


# 장점 :Saturate 되지 않아서 어디서나 정보를 갖는다. ( Good)
# 컴퓨터 연산이 효율적이다.( Good)
# 시그모이드, tanh 함수보다 수렴이 빠르다. ( Good)
# 단점 : knoct out 문제
ReLU의 문제점은?
#ReLU의 문제점은 입력값이 0보다 작을 때, 함수 미분값이 0이 되는 약점이 있습니다.
예제
## W,b 
W1 = tf.Variable(tf.random_normal(shape=[784,256]), name='weight1') 
b1 = tf.Variable(tf.random_normal(shape=[256]), name='bias1') 

# relu!!!!!! 사용!!!!! 
# 수식은 너무 복잡해서 주어진 함수 사용 
layer1 = tf.nn.relu(tf.matmul(X,W1) + b1) 

#sklearn.metrics
실제 데이터 중 맞게 예측한 데이터의 비율을 뜻한다 
matplotlib.pyplot
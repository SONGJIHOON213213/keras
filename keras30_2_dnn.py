from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
model = Sequential()                   #(N,3)
model.add(Dense(10, input_shape=(3,))) #(batch_size, input_dim)
model.add(Dense(units=15))             #출력(batch_size, units)
model.summary()
#2. 모델구성

# model.add(tf.keras.Input(shape=(16,)))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dense(32))
# model.output_shape
# (None, 32)

# 입력 형태

# 모양이 있는 ND 텐서: (batch_size, ..., input_dim). 가장 일반적인 상황은 모양이 있는 2D 입력입니다 (batch_size, input_dim).

# 출력 형태

# 모양이 있는 ND 텐서: (batch_size, ..., units). 예를 들어, 모양이 있는 2D 입력의 경우 (batch_size, input_dim)출력은 모양이 됩니다 (batch_size, units).

# Input shape

# N-D tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).

# Output shape

# N-D tensor with shape: (batch_size, ..., units). For instance, 
# for a 2D input with shape (batch_size행, input_dim열), the output would have shape (batch_size, units).



import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_train)
print(y_train)
print(x_train[4])
print(y_train[4])
import matplotlib.pyplot as plt
plt.imshow(x_train[100])
plt.show()
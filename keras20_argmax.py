import numpy as np

a = np.array([[1,2,3,],[6,4,5],[7,9,2],[3,2,1],[2,3,1]])
print(a)
print(a.shape)
print(np.argmax(a))
print(np.argmax(a, axis=0)) #[2 0 1 0 1]0은 행이야 그래서 행끼리 비교
print(np.argmax(a, axis=1))# [2 0- 1 0 1]1 열
print(np.argmax(a, axis=1))# [2 0 1 0 1]-1 가장마지막값
#가장 마지막 축 이건 2차원 가장 마지막축은 1차원
#그래서 -1을 쓰면 이 데이터는 1과 동일
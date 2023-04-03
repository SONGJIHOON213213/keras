import numpy as np

aaa = np.array([1,2,3])

bbb = aaa

bbb[0] = 4 
print(bbb)  # [4,2,3]
print(aaa) #aaa는 bbb의값을 참고하는거다 그래서 bbb값을 바꾸면 aaa의 값이 바뀐다.

print("==============================================")
ccc = aaa.copy() #새로운 메모리 구조 생성
ccc[1] = 7

print(ccc)
print(aaa)

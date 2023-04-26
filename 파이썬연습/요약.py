

# Python을 사용하고 있다면, input 대신 sys.stdin.readline을 사용할 수 있다. 
# 단, 이때는 맨 끝의 개행문자까지 같이 입력받기 때문에 문자열을 저장하고 싶을 경우 .rstrip()을 추가로 해 주는 것이 좋다.
# 또한 입력과 출력 스트림은 별개이므로, 테스트케이스를 전부 입력받아서 저장한 뒤 전부 출력할 필요는 없다. 테스트케이스를 하나 받은 뒤 하나 출력해도 된다.
# 자세한 설명 및 다른 언어의 경우는 이 글에 설명되어 있다.
# 이 블로그 글에서 BOJ의 기타 여러 가지 팁을 볼 수 있다.

# 입력
# 첫 줄에 테스트케이스의 개수 T가 주어진다. T는 최대 1,000,000이다. 다음 T줄에는 각각 두 정수 A와 B가 주어진다. A와 B는 1 이상, 1,000 이하이다.
# 출력
# 각 테스트케이스마다 A+B를 한 줄에 하나씩 순서대로 출력한다.
# 예제 입력 1 
# 5
# 1 1
# 12 34
# 5 500
# 40 60
# 1000 1000
# 예제 출력 1 
# 2
# 46
# 505
# 100
# 2000

#1
for i in[*open(0)][1:]:print(sum(map(int,i.split()))) 
#2
import sys

for line in sys.stdin.readlines()[1:]:
    a, b = map(int, line.split())
    print(a + b)
    
import sys 
for line in sys.stdin.readlines()[1:]:
    a,b = map(int, line.split())
    print(a + b)

import sys
for line in sys.stdin.readlines()[1:]:
    a,b = map(int, line.split())
    print(a + b) 
    
import sys 

for line in sys.stdin.readlines()[1:]:
    a,b = map(int,line.split())
    print(a + b) 


T = int(input())  

for i in range(C):
    A, B = map(int, input().split()) 
    print(f"Case #{i+1}: {A+B}")
    
T = int(input())

for i in range(1, T+1):
    A,B = map(int, input()split())
    print("Case #%d: %d + %d = %d" % (i, A, B, A+B))
    
    
# f-string은 파이썬 3.6 버전 이후 추가된 문자열 포맷팅 방법 중 하나로, 
# 문자열 내에 중괄호({})로 변수나 표현식을 간단하게 삽입하여 
# 문자열을 만들 수 있게 해줍니다. f는 format string의 약자이며, 
# 문자열 앞에 f를 붙여서 사용합니다.

# 예를 들어, 변수 x와 y가 주어졌을 때, "x + y = 5"
# 와 같은 문자열을 만들고 싶다면 다음과 같이 f-string을 사용할 수 있습니다.

# python
# Copy code
# x = 2
# y = 3
# result = f"{x} + {y} = {x + y}"
# print(result) # "2 + 3 = 5" 출력 


testcase = int(input()) 

for i in range(testcase):
    print(f"case #{i+1}:{A+B}")

testcase = int(input())

for i in range(testcase):
    A,B = map(int,input().split())
    print(f"case# {i+1}: {A+B}")
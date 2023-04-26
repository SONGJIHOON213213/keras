# while True:
#     try:
#         A, B = map(int, input().split())
#         if A == 0 and B == 0:
#             break
#         print(A + B)
#     except:
#         break 
N = int(input())
for i in range(1 , N+1):
    stars = '*' * i
    spaces = ' ' * i 
    print(f'{spaces}{stars}')  

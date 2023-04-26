coffee = 10
while True: 
    money = int(input("돈넣어 색갸:"))
    if money == 300:
        print("커피를 줍니다.")
        coffee = coffee -1 
    elif money > 300:
        coffee = coffee -1 
        print("커피주고 거스름돈%d를줍니다.")
    else:
        print("돈돌려주고 커피없음")
        print("남은커피양은 %d개입니다."% coffee) 
    if coffee == 0:
        print("커피 다떨어져서 판매중지") 
        break
import numpy as np
from tensorflow.keras.preprocessing import image #전처리preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator #전처리preprocessing

train_datagen = ImageDataGenerator(
    rescale=1./255,# 0~1 사이로 나눈다는거는 정규화 한다는 소리 노멀라이제이션,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode = 'nearest',
) #데이터증폭내용 

#테스트데이터는 평가데이터라서 증폭할필요가없다.  
    
test_datagen = ImageDataGenerator(
    rescale=1./255,    
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',
    target_size=(200,200),  #200에 200으로 확대또는축소
    batch_size=5,  
    class_mode='binary', #0바이너리 넣으면 0과1만됨
    color_mode='grayscale',
    shuffle=True
)
xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(200,200),  #200에 200으로 확대또는축소
    batch_size=5,
    class_mode='binary', #0바이너리 넣으면 0과1만됨
    color_mode='grayscale',
    shuffle=True,
)
print(xy_train)

print(xy_train[0])               # 32

print((xy_train[0]))             # 2      # 엑스 다섯개 들어가있다.               

print(xy_train[0][0].shape)            # [1. 0. 0. 0. 1.]               
print(xy_train[0][1].shape)            # [1. 0. 0. 0. 1.]               

print(xy_train[0][0].shape)            # [1. 0. 0. 0. 1.]               
print(xy_train[0][1].shape)            # [1. 0. 0. 0. 1.]               

print("################################################################")
print(type(xy_train[0]))
print(type(xy_train[0][1]))





#파이썬 이터레이터(Iterator)는 시퀀스(sequence) 데이터를 순회(iterate)하는 객체입니다. 
# 이터레이터는 시퀀스 데이터의 각 요소를 차례로 접근할 수 있는 방법을 제공합니다. 
# 이터레이터 객체는 일반적으로 for 루프를 사용하여 각 요소에 대한 작업을 수행하는 데 사용됩니다.
# 이터레이터는 두 가지 메서드를 구현해야 합니다. 첫 번째 메서드는 iter()이며, 
# 이는 이터레이터 객체 자체를 반환합니다. 두 번째 메서드는 next()이며, 이는 다음 요소를 반환합니다. 
# next() 메서드는 이터레이터가 마지막 요소에 도달한 경우 StopIteration 예외를 발생시키는 역할도 합니다.
# 이터레이터는 메모리 사용량을 최소화하면서 대용량 데이터셋을 처리하는 데 유용합니다. 
# 이터레이터는 데이터셋을 한 번에 읽지 않고 필요한 때에만 읽기 때문에, 매우 큰 데이터셋을 다룰 때 성능상 이점이 있습니다. 
# 예를 들어, 파일의 각 라인을 처리하거나, 데이터베이스 테이블의 각 행을 처리하는 데 이터레이터를 사용할 수 있습니다.
# 이터레이터 의 예시 만들어줘
# 아래는 파이썬에서 간단한 이터레이터 예시입니다.

# python
# Copy code
# class SquaresIterator:
#     def __init__(self, n):
#         self.i = 0
#         self.n = n

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.i >= self.n:
#             raise StopIteration
#         result = self.i ** 2
#         self.i += 1
#         return result

# squares = SquaresIterator(5)
# for square in squares:
#     print(square)

# self.current는 현재 이터레이터가 반환하는 값의 상태를 나타냅니다. 이터레이터는 __init__() 메서드에서 current 변수를 초기화하고, __
# next__() 메서드에서 이 값을 사용하여 다음 값을 반환합니다. 각 요소를 순회하면서 current 변수가 증가하므로, 
# 이터레이터는 항상 다음 값을 반환할 수 있도록 current 변수를 업데이트합니다. 
# 이터레이터는 __next__() 메서드가 호출될 때마다 current 변수를 증가시키고, 
# 새로운 값을 반환합니다. 이러한 방식으로, 이터레이터는 순서대로 데이터의 각 요소를 반환하게 됩니다.


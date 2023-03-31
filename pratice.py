import numpy as np
from tensorflow.keras.preprocessing import image #전처리preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator #전처리preprocessing
import time
from sklearn.model_selection import train_test_split 

start = time.time()  # 시작 시간 저장
path = 'd:/study_data/_save/hourse_or_human/'

test_datagen = ImageDataGenerator(rescale=1./255)

x = np.load(path + 'keras55x_1_train.npy')
y = np.load(path + 'keras55y_1_train.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True
)

xy_train = test_datagen.flow_from_directory(
    'd:/study_data/_data/hourse_or_human/human/',
    target_size=(100,100),  #200에 200으로 확대또는축소
    batch_size=500, 
    class_mode='binary', #0바이너리 넣으면 0과1만됨
    color_mode='grayscale',
    # color_mode='rgb',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/hourse_or_human/hourse/',
    target_size=(100,100),  #200에
)
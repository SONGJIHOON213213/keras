from PIL import Image
import os

# 이미지를 저장할 폴더 경로
folder_path = "d:/study_data/_data/dog/golden_retriever"

# 폴더 내 모든 이미지 파일 이름 가져오기
image_names = os.listdir(folder_path)

# 폴더 내 모든 이미지를 흑백으로 변환
for image_name in image_names:
    # 이미지 열기
    image_path = os.path.join(folder_path, image_name)
    image = Image.open(image_path)

    # 흑백으로 변환
    gray_image = image.convert("L")

    # 새로운 파일명 생성
    new_image_name = "gray_" + image_name

    # 흑백 이미지 저장
    gray_image.save(os.path.join(folder_path, new_image_name))
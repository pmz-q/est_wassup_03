import os
import random
import shutil

# data 폴더의 하위 폴더 목록을 가져옵니다.
emotions = os.listdir("../data")

# 각 하위 폴더의 `/raw/validation` 안에 있는 이미지 경로 목록을 저장합니다.
image_paths = {}
for emotion in emotions:
    image_paths[emotion] = []
    if emotion == "happy":
        for filename in os.listdir(os.path.join("../data", emotion, "raw", "train", "EMOIMG_기쁨_TRAIN_01")):
            if not filename.endswith(".zip"):
                image_paths[emotion].append(os.path.join("../data", emotion, "raw", "train", "EMOIMG_기쁨_TRAIN_01", filename))
    else:
        for filename in os.listdir(os.path.join("../data", emotion, "raw", "train")):
            if not filename.endswith(".zip"):
                image_paths[emotion].append(os.path.join("../data", emotion, "raw", "train", filename))

# 각 하위 폴더에서 랜덤으로 1%의 이미지 경로를 선택합니다.
selected_image_paths = {}
for emotion in emotions:
    selected_image_paths[emotion] = random.sample(image_paths[emotion], int(len(image_paths[emotion]) * 0.01))

# 새로운 폴더를 만들어 선택된 이미지를 해당 폴더로 복사합니다.
target_folder = "../data/selected_images_0.01"
os.makedirs(target_folder, exist_ok=True)

for emotion, paths in selected_image_paths.items():
    for path in paths:
        # 이미지 파일 이름만 추출
        image_filename = os.path.basename(path)
        # 대상 폴더에 해당 감정 폴더 생성 (해당 폴더가 없는 경우)
        target_emotion_folder = os.path.join(target_folder, emotion)
        os.makedirs(target_emotion_folder, exist_ok=True)
        # 이미지 복사
        shutil.copy(path, os.path.join(target_emotion_folder, image_filename))

# 결과 확인
print("Images copied to:", target_folder)

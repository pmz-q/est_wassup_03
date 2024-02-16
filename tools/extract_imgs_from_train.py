import os
import random
import shutil

# 샘플 뽑고싶은 퍼센티지
PERCENT = 0.1

# data 폴더의 하위 폴더 목록을 가져옵니다. (test_set 제외)
emotions = [item for item in os.listdir("../data") if item != "test_set"]

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

label_paths = {}
for emotion in emotions:
    for filename in os.listdir(os.path.join("../data", emotion, "labeled", "train")):
        label_paths[emotion] = os.path.join("../data", emotion, "labeled", "train", filename)

# 각 하위 폴더에서 랜덤으로 10%의 이미지 경로를 선택합니다.
selected_image_paths = {}
for emotion in emotions:
    selected_image_paths[emotion] = random.sample(image_paths[emotion], int(len(image_paths[emotion]) * PERCENT))

# 새로운 폴더를 만들어 선택된 이미지를 해당 폴더로 복사합니다.
target_folder = os.path.join("../work/selected_images_", PERCENT) 
os.makedirs(target_folder, exist_ok=True)

for emotion, paths in selected_image_paths.items():
    for path in paths:
        # 이미지 파일 이름만 추출
        image_filename = os.path.basename(path)
        # 대상 폴더에 해당 감정 폴더 생성 (해당 폴더가 없는 경우)
        target_emotion_image_folder = os.path.join(target_folder, "images", emotion)
        os.makedirs(target_emotion_image_folder, exist_ok=True)
        # 이미지 복사
        shutil.copy(path, os.path.join(target_emotion_image_folder, image_filename))


for emotion, paths in label_paths.items():
    label_filename = os.path.basename(paths)
    target_emotion_label_folder = os.path.join(target_folder, "labels", emotion)
    os.makedirs(target_emotion_label_folder, exist_ok=True)
    shutil.copy(paths, os.path.join(target_emotion_label_folder, label_filename))

print("Images copied to:", target_folder)

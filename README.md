# est_wassup_03

# Quick Start
## 1. Data ref
- 원천데이터: [AI Hub: 한국인 감정인식을 위한 복합 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=82)
```
/images
    /anger
    /happy
    /sad
    /emb
    ...
/labels
    /anger
    /happy
    /sad
    /emb
    ...
/test_set
    /anger
    /anxiety
    /embarrass
    /happy
    /normal
    /pain
    /sad
    /anger.json
    /anxiety.json
    /embarrass.json
    /happy.json
    /normal.json
    /pain.json
    /sad.json
```
## 2. Data pre-processing
### 2-1. tools/sample_data_into_yolo_structure.py
원천데이터로부터 랜덤으로 특정 퍼센테지의 데이터를 추출합니다.
- `--selection-ratio`: float: 추출하기를 원하는 데이터의 퍼센테지
- `--src-data-path`: 원천데이터의 데이터 경로
- `--dst-data-path`: 샘플링 된 데이터를 저장할 경로
```
/images
    /train
        /anger
        /happy
        /sad
        /emb
        ...
    /test
        /anger
        /happy
        /sad
        /emb
        ...
/labels
    /train
        /anger
        /happy
        /sad
        /emb
        ...
    /test
        /anger
        /happy
        /sad
        /emb
        ...
```
### 2-2. tools/create_feature_dataset.py
한글 파일명들을 영문으로 변환합니다. 모든 파일명을 uuid4 로 변환합니다.
Coco format 으로 라벨을 저장합니다. (감정당 annotation.json)
여기서 나온 결과 데이터를 원본데이터라고 부릅니다.
- `--src-dir-path`: 샘플링 된 데이터의 경로
- `--dst-dir-path`: 변환된 데이터를 저장할 경로
```
/images
    /train
        /anger
          /uuid4.jpg
          ...
        /happy
        /sad
        /emb
        ...
    /test
        /anger
        /happy
        /sad
        /emb
        ...
/labels
    /train
        /happy
            /annotation.json
        /sad
        ...
    /test
        /happy
            /annotation.json
        /sad
        ...
```
### 2-3. tools/create_yolo_detection_dataset.py
yolo detection model 을 돌리기 위한 데이터셋을 생성합니다.
- `--src-data-path`: 원본 데이터의 경로
- `--dst-data-path`: yolo detection model dataset 을 저장할 경로
```
/images
    /train
      /uuid4.jpg
    /test
/labels
    /train
      /uuid4.txt
    /test
```
### 2-4. tools/split_train_val.py
원본 데이터셋 혹은 yolo detection dataset 의 trainset 을 특정 비율로 train set 과 validation set 으로 나눕니다.
yolo detection dataset 의 경우, train set, validation set 그리고 test set 의 경로를 지정해주는 `yolo-dataset.yaml` 파일도 함께 생성합니다.
- `--annot-format`: ['coco', 'yolo']: 어노테이션 포맷이 coco 인지, 혹은 yolo 형태인지를 지정합니다.
- `--task`: ['classification', 'detection']: 데이터셋이 classification 을 위한 데이터셋인지, 혹은 detection 을 위한 데이터셋인지 를 지정합니다.
- `--src-root-path`: split 하고자 하는 dataset 의 경로
- `--dst-root-path`: 결과물을 저장할 경로
- `--train-ratio`: float: 원본 데이터를 train 과 validation 으로 나눌 때, train 의 비율
```
<COCO and YOLO classification data format> 

/images
    /train
        /anger
        /happy
    /test
        /anger
        /happy
/labels
    /train
        /anger
        /happy
    /test
        /anger
        /happy

<YOLO Detection data format>

/images
    /train
    /test
/labels
    /train
    /test
```
### 2-5. tools/detect_face.py
yolo detection weight 를 통해 이미지에서 얼굴 이미지만 crop 하여 데이터셋으로 저장합니다.
이러한 데이터셋을 cropped dataset 이라고 부릅니다.
만약, model 이 face 를 detect 하지 못할 경우, 해당 이미지는 데이터셋에서 제외됩니다.
- `--dir-option`: ['train', 'val', 'test']: crop 할 데이터셋이 split 된 데이터셋인지 아닌지에 따라서 dir-option 을 지정해주면 됩니다.
- `--src-data-path`: crop 할 데이터셋의 경로
- `--dst-data-path`: cropped dataset 이 저장될 경로
- `--weights-path`: yolo detection weight 의 경로
- `--target-size`: tuple: crop 될 이미지의 크기
- `--padding-option`: ['yolo', 'custom']: crop 된 이미지가 target-size 를 맞추기 위해 나머지 공간들을 채울 방범. yolo 의 경우, 배경을 추가로 넣습니다. custom 의 경우 검은색으로 채웁니다.
- `--padding-scale`: float: target-size 안에서 padding 의 비율을 높이고자 할 때, 기본으로 0.8을 주고 있고, 0.7 이 recommended 입니다.
### 2-6. tools/sync_coco_annot_with_imgs.py
images 폴더의 정보를 읽고 coco annotation 을 업데이트 합니다.
- `--dir-option`: ['train', 'val', 'test']: sync 할 데이터셋이 split 된 데이터셋인지 아닌지에 따라서 dir-option 을 지정해주면 됩니다.
- `--data-path`: sync 할 데이터셋의 경로
## 3. Configs
Train 과 inference 를 실행하기 위해선 config 파일을 생성해야 합니다.
config 파일 샘플은 configs 폴더 내에 저장되어 있습니다.
### 3-1. info
- `project_name`: 프로젝트의 이름: 학습 후 결과물을 저장할 때 사용됩니다.
- `model_name`: 모델명: resnet50, resnet101, resnet152, yolo 중 하나를 선택하면 됩니다.
- `model_task`: classification 혹은 detection 중 하나 선택하면 됩니다.
- `data_root_dir`: 학습에 사용될 데이터셋의 경로
### 3-2. train
- `epochs`: epoch
- `lr_scheduler`: coslr or steplr: NOTE: yolo 모델의 경우 coslr 만 지원합니다.
- `lr_scheduler_params`: lr scheduler 에 kwargs 로 들어갑니다.
- `batch`: batch-size
- `imgsz`: int: 정사각형 이미지
- `device`: int: gpu device id: # if gpu is not available, enter cpu or mps
- `optimizer`: optimizer
- `optimizer_params`: optimizer params
- `pretrained`: 학습에 사용될 weigth 경로
- `seed`: random seed
- `dropout`: drop out ratio or 0.0 for no dropout
### 3-3. infer
- `src_dir`: inference 를 진행할 이미지들이 담긴 폴더
- `pretrained`: infer 를 진행할 weight 의 경로
- `infer_task`: ['classification', 'detection']: weight 의 모델 task 에 따라서 결정됩니다.
- `save_mode`: ["gray", "color", "heatmap"]: dlib 를 통한 landmark 정보 이미지 추출 시 사용됩니다. 컬러, 흑백 혹은 히트맵 저장 가능합니다.
```yaml
info: # 프로젝트의 정보
  project_name: project_1
  model_name: resnet50
  model_task: classification
  data_root_dir: abolute path
train:
  epochs: 20
  lr_scheduler: coslr
  lr_scheduler_params:
    # step_size: 2 # steplr params
    # gamma: 0.1 # steplr params
    # last_epoch: -1 # steplr params
    T_max: 10
    eta_min: 0.000001
  batch: 16
  imgsz: 224
  device: 0 
  optimizer: adamw # available optimizer = ['sgd', 'adam', 'adamw']
  optimizer_params:
    lr: 0.001
    weight_decay: 0.01  
  pretrained: abolute path # pretrained weight if wanted
  seed: 2024
  dropout: 0.3 # 0.0 for no dropout
  conf: # object detection confidence threshold
  loss_fn: cross_entropy # ['cross_entropy']
infer:
  src_dir: abolute path
  pretrained: abolute path
  infer_type: "dlib"
  infer_task: "detection"
  save_mode: ["gray", "color", "heatmap"]

```
## 4. Train
### 4-1. yolo detection
yolo detection 모델로 학습을 진행합니다.
config 에 `model_task` 가  `detection` 으로 지정되어있어야 합니다.
config 에 `model_name` 이 `yolo` 로 지정되어있어야 합니다.
`est_wassup_03/run.py` 를 실행합니다.
- `-run`: ["train", "infer", "eval"]: "train"
- `-type`: ["yolo", "coco"]
- `-cfg`: config 의 경로
### 4-2. yolo classification
yolo classification 모델로 학습을 진행합니다.
config 에 `model_task` 가  `classification` 으로 지정되어있어야 합니다.
config 에 `model_name` 이 `yolo` 로 지정되어있어야 합니다.
`est_wassup_03/run.py` 를 실행합니다.
- `-run`: ["train", "infer", "eval"]: "train"
- `-type`: ["yolo", "coco"]: "yolo"
- `-cfg`: config 의 경로
### 4-3. resnet50,resnet101,resnet152 classification
resnet classification 모델로 학습을 진행합니다.
config 에 `model_task` 가  `classification` 으로 지정되어있어야 합니다.
config 에 `model_name` 이 `resnet50,resnet101,resnet152` 중 하나로 지정되어있어야 합니다.
`est_wassup_03/run.py` 를 실행합니다.
- `-run`: ["train", "infer", "eval"]: "train"
- `-type`: ["yolo", "coco"]: "coco"
- `-cfg`: config 의 경로
### 4-4. posterv2 classification
4-4. posterv2 classification
posterv2 classification 모델로 학습을 진행합니다.
est_wassup_03/core/models/poster/main_poster.py 를 실행합니다.
- `--data` : 데이터 소스 경로
- `--data_type`: 데이터셋 타입 (augmentation 방식에 차이가 있습니다)
- `--checkpoint_path`: 데이터셋마다 학습한 weights 저장 경로
- `--best_checkpoint_path`: 3번의 best weights
- `-j`, `--workers`: dataloader에 사용하는 worker 개수
- `--epochs`
- `--start-epoch`: resume용 manual start epoch
- `-b`, `--batch-size`
- `--optimizer`: adamw, adam, sgd 중에서 선택
- `--lr`: learning rate
- `--momentum`
- `--wd`, `--weight-decay`
- `-p`, `--print-freq`
- `-e`, `--evaluate`: 테스트 셋에다가 evaluate할때 쓰는 모드
- `--beta`: argparser에만 있고 main함수에선 사용 안됨
- `--gpu`: gpu index
## 5. Inference
### 5-1. dlib
face landmark 를 detection 하기위한 기능입니다.
다른 inference option 들과 다르게 coco annotation 데이터셋 구조의 데이터셋에서 가능합니다.
`.dat` 파일이 필요합니다. # .dat file to predict landmarks # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
config 에 `infer` 안에 `pretrained` 를 `.bat` 파일의 경로로 지정해줍니다.
config 에 `infer_type` 이  `dlib` 으로 지정되어있어야 합니다.
config 에 `infer_task` 가 `detection` 으로 지정되어있어야 합니다.
`est_wassup_03/run.py` 를 실행합니다.
- `-run`: ["train", "infer", "eval"]: "infer"
- `-type`: ["yolo", "coco"]: "coco"
- `-cfg`: config 의 경로
### 5-2. models
config 에 `infer` 안에 `pretrained` 를 weight 의 경로로 지정해줍니다.
config 에 `infer_type` 이  모델명으로 지정되어있어야 합니다.
config 에 `infer_task` 가 `detection` 으로 지정되어있어야 합니다.
`est_wassup_03/run.py` 를 실행합니다.
- `-run`: ["train", "infer", "eval"]: "infer"
- `-type`: ["yolo", "coco"]
- `-cfg`: config 의 경로
## 6. Results
### 6-1. yolo
- `est_wassup_03/model_task/project_name/run_type/{1부터 순차적으로 증가하는 숫자}/` 에서 결과를 확인할 수 있습니다.
### 6-2. non-yolo
- `est_wassup_03/model_task/project_name/run_type/{1부터 순차적으로 증가하는 숫자}/` 에서 결과를 확인할 수 있습니다.
- Train extra results
  - tensorboard 를 활용하여 loss 그래프, accuracy 그래프, learning rate 그래프 등 확인할 수 있습니다.
  - results 폴더를 기준으로 tensorboard 실행하시면 됩니다.
  - `tensorboard --logdir={path_to_results}`




<div align="center">car detection for fgvc using YOLOv5</div>


<div align="center"> This code is from <a href=https://github.com/ultralytics/yolov5>YOLOv5</a> </div>

<details>
<summary>preprocessing</summary>  

```bash  
cd preprocess
python make_yaml.py # 학습을 위한 yaml파일 생성
python preprocess.py  # input 't' for training set and 'v' for validation set  
```  

preprocess.py를 실행했을 때 command창에 t 입력,
처리 완료되면 다시 한번 preprocess.py를 실행시킨 후 v를 눌러준다.
</details>

<details>
<summary>pseudo labeling</summary>
데이터셋의 annotation label이 부정확할 때 사용.

```bash
cd pseudo_labeling
python pseudo_car_labeling.py # 필요시 인자값 설정, 완료되면 runs/detect 디렉토리에 자동차 전체에 대한 pseudo label 파일 생성 
python label_mergy.py # dataset 디렉토리안의 training/labels안에 merge_labels파일 생성. 이 디렉토리를 training/labels(merge_labels이름변경)로 변경해줘야함
```
</details>

<details>
<summary>train</summary>
train, test는 official tutorial 참고

```bash
cd ..
python train.py --cfg models/yolov5s.yaml --weights yolov5s.pt --data data/cctv_justcar.yaml --epochs 30 --batch-size 16 --name epoch30_batch16 # 예시 command 
```
</details>
# Fine-grained recognition (car class)


## <div align="center">Model flow</div>

    Input image -> train YOLOv5s(detection) -> pseudo labeling  
                                                        ㄴ> retrain Yolov5s -> image crop -> TransFG -> Prediction



### 원본 이미지   
![alt text](assets/C-220805_04_CR02_01_A0075.jpg)   

### pseudo label 적용 이미지 & annotation
초록 bbox annotation이 원래의 bbox annotation  
빨강 bbox annotation이 pseudo label 적용한 bbox annotation

![alt text](assets/image.png)

### image crop
<div align="center" style="width:image width px;">
  <img  src="assets/C-220721_15_CR12_03_A0993.jpg" width=350 alt="샘플1">
  <img  src="assets/C-220806_14_CR13_05_A2253.jpg" width=350 alt="샘플2">
</div>

### Results
<div align="center" style="width:image width px;"> 
  <img  src="assets/transfg결과1.PNG" width=350 alt="결과1">
  <img  src="assets/transfg결과2.PNG" width=350 alt="결과2">
  <img  src="assets/transfg결과3.PNG" width=350 alt="결과3">
</div>
<br/>


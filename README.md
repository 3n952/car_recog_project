# Fine-grained recognition (car class)


## <div align="center">Model flow</div>

    Input image -> train YOLOv5s(detection) -> pseudo labeling  
                                                        ㄴ> retrain Yolov5s -> image crop -> TransFG -> Prediction



### 원본 이미지   
![alt text](assets\C-220805_04_CR02_01_A0075.jpg)   

### pseudo label 적용 이미지 & annotation
초록 bbox annotation이 원래의 bbox annotation  
빨강 bbox annotation이 pseudo label 적용한 bbox annotation

![alt text](assets\image.png)

### image crop
![alt text](assets\C-220721_15_CR12_03_A0993.jpg)
![alt text](assets\C-220806_14_CR13_05_A2253.jpg)

### Results

![alt text](assets\transfg결과1.PNG)
![alt text](assets\transfg결과2.PNG)
![alt text](assets\transfg결과3.PNG)
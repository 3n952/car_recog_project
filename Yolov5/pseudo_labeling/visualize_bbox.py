import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import random
from glob import glob 
import os
import ast

def visualize_original_pseudo(root_dir, is_train=True):

    # 랜덤한 이미지에 대한 원래의 bbox, pseudo bbox 시각화


    if is_train:
        json_path = sorted(glob(os.path.join(root_dir, 'Training', 'annotations','*.json')))
        # pseudo_path = sorted(glob(os.path.join(root_dir, 'Training_pseudo', 'labels', '*.txt')))
        img_path = os.path.join(root_dir, 'Training', 'images')
        txt_path =  os.path.join(root_dir, 'Training_pseudo', 'labels')
    else:
        json_path = sorted(glob(os.path.join(root_dir,'Validation', 'annotations', '*.json')))
        # pseudo_path = sorted(glob(os.path.join(root_dir, 'Validation_pseudo', 'labels', '*.txt')))
        img_path = os.path.join(root_dir, 'Validation', 'images')
        txt_path =  os.path.join(root_dir, 'Validation_pseudo', 'labels')
        
    random_num = [random.randint(0, len(json_path)) for _ in range(5)]

    for idx in random_num:
        #bbox 초기화
        og_bbox = []
        pseudo_bbox = []

        json_path2 = json_path[idx]
        with open(json_path2, 'rb') as jsfile:
            json_data = json.load(jsfile)
            fname = json_data['Source Data Info']['source_data_id']
            print(fname)
            fname2 = fname+'.jpg'
            fname3 = fname+'.txt'

            # annotation bbox 
            for i in range(len(json_data['Learning Data Info']['annotations'])):
                bbox = json_data['Learning Data Info']['annotations'][i]['coord']
                if isinstance(bbox, str):
                    bbox = ast.literal_eval(bbox)

                og_bbox.append(bbox)
        
        try:
            pseudo_path = os.path.join(txt_path, fname3)
        except:
            print('try again')

        with open(pseudo_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_split = line.split()
                bbox = [float(line_split[1]),float(line_split[2]),float(line_split[3]),float(line_split[4])]
                pseudo_bbox.append(bbox)

        image_path = os.path.join(img_path, fname2)

        # cv2 이미지 불러오기 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 이미지를 불러오므로 RGB로 변환

        image_height, image_width, _ = image.shape

        # 이미지에 바운딩 박스를 시각화
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # visualize pseudo bbox
        for bbox in pseudo_bbox:
            x_center, y_center, width, height = bbox
                
            # 정규화된 좌표를 픽셀 좌표로 변환
            x_center_pixel = x_center * image_width
            y_center_pixel = y_center * image_height
            width_pixel = width * image_width
            height_pixel = height * image_height
                
            # 좌상단 좌표 계산
            x_min = x_center_pixel - width_pixel / 2
            y_min = y_center_pixel - height_pixel / 2
                
            # 사각형(Rectangle) 추가
            rect = patches.Rectangle((x_min, y_min), width_pixel, height_pixel, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_min, y_min - 10, 'pseudo', color='red', fontsize=8, weight='bold')


        # visualize annotation bbox
        for bbox in og_bbox:
            x_center, y_center, width, height = bbox

            x_center = (x_center + (x_center + width)) / 2
            y_center = (y_center + (y_center + height)) / 2

            # 정규화된 좌표를 픽셀 좌표로 변환
            # x_center_pixel = x_center * image_width
            # y_center_pixel = y_center * image_height
            # width_pixel = width * image_width
            # height_pixel = height * image_height
                
            # 좌상단 좌표 계산
            x_min = x_center - width / 2
            y_min = y_center - height / 2
                
            # 사각형(Rectangle) 추가
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_min, y_min - 10, 'og', color='red', fontsize=8, weight='bold')

        plt.show()

def visualize_label(img_dir, label_dir):

    # 랜덤하게 해당 디렉토리 라벨링 시각화

    label_path = sorted(glob(os.path.join(label_dir,'*.txt')))
    random_num = [random.randint(0, len(label_path)) for _ in range(10)]

    for idx in random_num:
        #bbox 초기화
        bbox = []

        with open(label_path[idx], 'r', encoding='utf-8') as file:
            for line in file:
                line_split = line.split()
                bbox_ = [float(line_split[1]),float(line_split[2]),float(line_split[3]),float(line_split[4])]
                bbox.append(bbox_)

        fname = os.path.basename(label_path[idx][:-4]) + '.jpg'
        image_path = os.path.join(img_dir, fname)

        # cv2 이미지 불러오기 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 이미지를 불러오므로 RGB로 변환

        image_height, image_width, _ = image.shape

        # 이미지에 바운딩 박스를 시각화
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # visualize pseudo bbox
        for box in bbox:
            x_center, y_center, width, height = box
                
            # 정규화된 좌표를 픽셀 좌표로 변환
            x_center_pixel = x_center * image_width
            y_center_pixel = y_center * image_height
            width_pixel = width * image_width
            height_pixel = height * image_height
                
            # 좌상단 좌표 계산
            x_min = x_center_pixel - width_pixel / 2
            y_min = y_center_pixel - height_pixel / 2
                
            # 사각형(Rectangle) 추가
            rect = patches.Rectangle((x_min, y_min), width_pixel, height_pixel, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_min, y_min - 10, 'car', color='red', fontsize=8, weight='bold')

        plt.show()

root_dir = r'D:\cctv_datasets_yolo\cm\cm_datasets'
img_dir = r'D:\cctv_datasets_yolo\cm\cm_datasets\Training\images'
label_dir = r"D:\cctv_datasets_yolo\cm\cm_datasets\Training\labels"

# 원래의 bbox 라벨링과 pseudo 라벨링을 동시에 시각화 할 때
#visualize_original_pseudo(root_dir=root_dir)

# 한 디렉토리 안의 bbox 시각화 할 때
visualize_label(img_dir=img_dir, label_dir=label_dir)
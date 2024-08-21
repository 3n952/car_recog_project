import cv2
import json
import random
from glob import glob 
import os
import ast


# box 1은 annotation data, box 2는 Pseudo label
def calculate_iou(box1, box2):
    # 바운딩 박스는 (x_center, y_center, width, height) 형식.
    x_center1, y_center1, width1, height1 = box1
    x_center2, y_center2, width2, height2 = box2

    # 첫 번째 바운딩 박스의 좌상단 및 우하단 좌표
    x_min1 = x_center1 - width2 / 2
    y_min1 = y_center1 - height2 / 2
    x_max1 = x_center1 + width2 / 2 
    y_max1 = y_center1 + height2 / 2

    # 두 번째 바운딩 박스의 좌상단 및 우하단 좌표
    x_min2 = x_center2 - width2 / 2
    y_min2 = y_center2 - height2 / 2
    x_max2 = x_center2 + width2 / 2
    y_max2 = y_center2 + height2 / 2

    # 교차 영역의 좌상단 및 우하단 좌표
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # 교차 영역의 너비와 높이
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)

    # 교차 영역의 면적
    inter_area = inter_width * inter_height

    # 각 바운딩 박스의 면적
    area1 = width1 * height1
    area2 = width2 * height2

    # 두 바운딩 박스의 합집합 영역 면적
    union_area = area1 + area2 - inter_area

    # IoU 계산
    iou = inter_area / union_area

    return iou

def label_merge(root_dir, is_train = True):

    if is_train:
        # txt format
        pseudo_label_path =  os.path.join(root_dir, 'Training_pseudo', 'labels')
        anno_label_path = os.path.join(root_dir, 'Training', 'labels')

    else:
        pseudo_label_path =  os.path.join(root_dir, 'Validation_pseudo', 'labels')
        anno_label_path = os.path.join(root_dir, 'Validation', 'labels')
    

    label_path = glob(os.path.join(pseudo_label_path, '*.txt'))

    # pseudo_txt_path = 'root_dir/Training_pseudo/labels/*.txt'
    for pseudo_txt_path in label_path:

        # anno_label_path = 'root_dir/Training/labels/*.txt'
        anno_txt_path = os.path.join(anno_label_path, os.path.basename(pseudo_txt_path))



        #bbox 초기화
        og_bbox = []
        pseudo_bbox = []
        merge_bbox = []

        with open(pseudo_txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_split = line.split()
                bbox = [float(line_split[1]),float(line_split[2]),float(line_split[3]),float(line_split[4])]
                pseudo_bbox.append(bbox)

        with open(anno_txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_split = line.split()
                bbox = [float(line_split[1]),float(line_split[2]),float(line_split[3]),float(line_split[4])]
                og_bbox.append(bbox)

        
        width, height = 3840. , 2160.

        # calculate iou for merge bbox 
        for p_bbox in pseudo_bbox:
            for o_bbox in og_bbox:
                iou = calculate_iou(o_bbox, p_bbox)
                if iou >= 0.8:
                    if not o_bbox in merge_bbox:
                        merge_bbox.append(o_bbox)
                        break
                elif iou == 0:
                    if not pseudo_bbox in merge_bbox:
                        merge_bbox.append(pseudo_bbox)
                else:
                    if not pseudo_bbox in merge_bbox:
                        merge_bbox.append(pseudo_bbox)
        
        fname2write = os.path.join(anno_label_path, 'pseudo_labels', os.path.basename(pseudo_txt_path))
        with open(fname2write, 'w') as anno:
            for bbox in range(merge_bbox):
                anno.write(f'0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')

           


root_dir = r'D:\cctv_datasets_yolo\cm\cm_datasets'
label_merge(root_dir, is_train = True)

import cv2
import json
import random
from glob import glob 
import os
import ast

def denormalize_bbox(bbox):

    image_width, image_height = float(3840), float(2160)
    x_center, y_center, width, height = bbox

    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height

    bbox = [x_center_pixel, y_center_pixel, width_pixel, height_pixel]

    return bbox

# box 1은 annotation data, box 2는 Pseudo label
def calculate_iou(box1, box2):
   
    # width = float(3840)
    # height = float(2160)

    # 바운딩 박스는 (x_center, y_center, width, height) 형식.
    box1 = denormalize_bbox(box1)
    box2 = denormalize_bbox(box2)

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
                pbbox = [float(line_split[1]),float(line_split[2]),float(line_split[3]),float(line_split[4])]
                pseudo_bbox.append(pbbox)

        with open(anno_txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_split = line.split()
                obbox = [float(line_split[1]),float(line_split[2]),float(line_split[3]),float(line_split[4])]
                og_bbox.append(obbox)

        
        #width, height = 3840. , 2160.

        # calculate iou for merge bbox 
        for p_bbox in pseudo_bbox:
            for o_bbox in og_bbox:
                iou = calculate_iou(o_bbox, p_bbox)
                if iou >= 0.68:
                    if not o_bbox in merge_bbox:
                        merge_bbox.append(o_bbox)
                        break
                else:
                    if not pseudo_bbox in merge_bbox:
                        merge_bbox.append(p_bbox)

        # 중복된 리스트를 제거하기 위해 set을 사용
        final_bbox = []
        seen = set()
    
        for lst in merge_bbox:
            # 리스트를 튜플로 변환하여 set에 추가할 수 있게 만듦
            lst_tuple = tuple(lst)
            if lst_tuple not in seen:
                seen.add(lst_tuple)
                final_bbox.append(lst)
        
        fname2write = os.path.join(anno_label_path, 'merge_labels', os.path.basename(pseudo_txt_path))
        with open(fname2write, 'w') as anno:
            for bbox in final_bbox:
                anno.write(f'0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')

           


root_dir = r'..\dataset'
# 파일 생성 경로: labels/pseudo_labels
label_merge(root_dir, is_train = False)

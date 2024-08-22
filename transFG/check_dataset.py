# preprocessing을 마치고 image 개수 label 개수 -> 경로에 맞게 잘 설정되었는지 확인

import os
from glob import glob
import csv
import pandas as pd


def check_dataset(img_root, csv_root):
    csv_file_num = 0
    img_file_num = 0

    for i in glob(os.path.join(csv_root, '*_x.csv')):
        df = pd.read_csv(i)
        csv_file_num += len(df)
    
    for j in glob(os.path.join(img_root, '*')):
        img_file_num += len(os.listdir(j))
    
    if csv_file_num == img_file_num:
        print('prepared to train!')
    else:
        print('not to prepare for training! check your datasets')
        print(f"imgs: {img_file_num}, paths: {csv_file_num}")
    

csv_root = r'C:\Users\QBIC\Desktop\workspace\car_model_recogize\transFG'
img_root = os.path.join(csv_root, r'datasets\custom')

check_dataset(img_root=img_root, csv_root=csv_root)





import pandas as pd
import os
from glob import glob
import shutil

# 기존 train_x.csv ..등등에 섞인 데이터에 unknown이 섞여있으면 포함시키지 않음. + 해당 이미지 삭제

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
relative_parent_dir = os.path.join(current_dir, '..')
parent_dir = os.path.abspath(relative_parent_dir)

def csv_filter(og_csv, new_csv):
    # CSV 파일을 DataFrame으로 불러오기
    df = pd.read_csv(og_csv)

    # 특정 열의 이름 지정 (예: 'column_name')
    column_name = 'label_'

    # 'unknown'이 포함된 행을 삭제
    df = df[~df[column_name].str.contains('unknown', case=False, na=False)]
    df = df[~df[column_name].str.contains(r'\(unknown\)', case=False, na=False)]

    # 변경된 DataFrame을 다시 CSV 파일로 저장 (필요시)
    df.to_csv(new_csv, index=False) 

def img_filter(dataset_dir):
    
    for dir in glob(os.path.join(dataset_dir, '*unknown*')):
        try:
            shutil.rmtree(dir)
        except: 
            pass


if __name__ == "__main__":

    for i in glob(os.path.join(parent_dir, '*_x.csv')):
        csv_filter(i, i[:-4] + '_new.csv')
    
    img_filter(os.path.join(parent_dir, r'datasets\custom'))
    

       
     
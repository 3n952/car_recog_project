import pandas as pd
import os
from glob import glob

# 기존 train_x.csv ..등등에 섞인 데이터에 unknown이 섞여있으면 포함시키지 않음.

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
            
    # 정수형 label링 올림차순
    # df_sorted = df.sort_values(by='label', ascending=True)

    # 삭제 시 불연속적인 정수 값을 채우기
    # previous_value = -1

    # for i in range(len(df_sorted)):
    #     current_value = df_sorted.iloc[i]['label']
    #     if current_value != previous_value + 1:
    #         df_sorted.at[df_sorted.index[i], 'label'] = previous_value + 1
    #     previous_value = df_sorted.iloc[i]['label']

    # 변경된 DataFrame을 다시 CSV 파일로 저장 (필요시)
    df.to_csv(new_csv, index=False)

if __name__ == "__main__":

    for i in glob(os.path.join(parent_dir, '*_x.csv')):
        csv_filter(i, i[:-4] + '_new.csv')
    

       
     
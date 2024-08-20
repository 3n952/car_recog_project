from preprocess import string2int

import pandas as pd
import ast
import json
from glob import glob
import os
import yaml

def class_categorized(json_dir):
    files = glob(os.path.join(json_dir, '*.json'))
    df = pd.DataFrame(columns=['file_name','car_class'])

    for file in files:

        fname_list = []
        with open(file, 'rb') as jsfile:
            json_data = json.load(jsfile)
            fname_list.append(json_data['Source Data Info']['source_data_id'])
        

            for i in range(len(json_data['Learning Data Info']['annotations'])):
                class_list = []
                class_list.append(json_data['Learning Data Info']['annotations'][i]['model_id'])
                

                new_df = pd.DataFrame(zip(fname_list, class_list), columns = ['file_name','car_class'])
                df = pd.concat([df, new_df], ignore_index= True)
    
    return df

def label_category_dict(df):
    df,_ = string2int(df)
    df_unique = df.drop_duplicates(subset=['label'])
    selected_columns = df_unique[['label', 'car_class']]
    data_dict = selected_columns.to_dict(orient='records')

    # 빈 딕셔너리 생성
    result_dict = {}

    # 리스트 인덱스를 순회하면서 각 딕셔너리의 값으로 새로운 딕셔너리 생성
    for item in data_dict:
        key = item['label']
        value = item['car_class']
        result_dict[key] = value
    
    return result_dict

def make_yaml_func(lc_dict):
    # YAML 파일로 저장
    with open('cctv(label-class).yaml', 'w') as file:
        yaml.dump(lc_dict, file, default_flow_style=False, allow_unicode=True)




if __name__ == '__main__':

    print('label - class 딕셔너리 형태의 yaml 파일 생성 시작')
    
    input_annodir = input('input your annotation directory:\n')
    new_df = class_categorized(input_annodir)
    new_dict = label_category_dict(new_df)

    make_yaml_func(new_dict)
    print('생성 완료')

    


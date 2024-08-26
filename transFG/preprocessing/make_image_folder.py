# encoding=utf8
import pandas as pd
import os
import shutil
import json
import time
from datetime import timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import ast
import json

from IPython.display import display 


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
relative_parent_dir = os.path.join(current_dir, '..')
parent_dir = os.path.abspath(relative_parent_dir)

start = time.time()
count = 0
df_all = pd.DataFrame()
data_root = os.path.join(parent_dir,'raw_datasets','라벨링데이터')
data_root2 = os.path.join(parent_dir,'raw_datasets','원천데이터')
road_3 = os.listdir(f'{data_root}\\차종외관인식') # 교차로, 이면도로, 접근로
for i in road_3:
    road_detail = os.listdir(f'{data_root}\\차종외관인식\\{i}')
    
    for j in tqdm(road_detail):
        road_num = os.listdir(f'{data_root}\\차종외관인식\\{i}\\{j}')
        
        for k in road_num:
            json_file_list = os.listdir(f'{data_root}\\차종외관인식\\{i}\\{j}\\{k}')
            
            for l in json_file_list:
                path = os.path.join(f'{data_root}\\차종외관인식\\{i}\\{j}\\{k}\\{l}')
                
                path = os.path.join(f'{data_root}\\차종외관인식\\{i}\\{j}\\{k}', l)
                with open(path,'r', encoding="UTF-8") as dc:
                    data = json.load(dc)
                    df = pd.DataFrame(data['Learning Data Info']['annotations'])
                    # 좌측 /없애기 + '/'를 '\\'로 교체 윈도우 기반아니면 .replace()삭제
                    df['path'] = os.path.join(data_root2, data['Learning Data Info']['path'].lstrip('/')).replace('/', '\\')
                    df['json_data_id'] = data['Learning Data Info']['json_data_id']
                    df_all = pd.concat([df_all, df], ignore_index=True)
                    count +=1

end = time.time()
print("Time elapsed: ", timedelta(seconds=end-start))
count

df_all = df_all.reset_index(drop=True)

df_all = df_all[df_all['model_id'] != 'Unknown']


df_all['json_data_id'] = df_all['path']+'\\'+df_all['json_data_id']

df_all['json_data_id'] = df_all['json_data_id']+'.jpg'

#coord => 좌측하단 x,y좌표 + width, height
def edit_coord(x):
    x[2] = x[0]+x[2]
    x[3] = x[1]+x[3]
    return x

#df_all['json_data_id'] = df_all['json_data_id'].apply(lambda x: x.replace('\\차종외관인식',''))
unique_model_list = df_all.drop_duplicates(['brand_id','model_id'])[['brand_id','model_id']]
unique_model_list.head()

ouput_path = os.path.join(parent_dir,'datasets/custom')


import os

def makedirs(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Failed to create the directory.")


start = time.time()
count = 0
for _,i in tqdm(df_all.iterrows()):
    try:
        count +=1
        if isinstance(i['coord'], str):
            i['coord'] = edit_coord(ast.literal_eval(i['coord']))

        i['coord'] = edit_coord(i['coord'])
        image1 = Image.open(os.path.join(f"{data_root2}\\차종외관인식", i['json_data_id']))

        image2 = image1.crop(i['coord'])
        folder_path = os.path.join(ouput_path,i['brand_id']+'#'+i['model_id'])
        makedirs(os.path.join(ouput_path,folder_path))
        image2.save(os.path.join(ouput_path,folder_path,i['json_data_id'].split('\\')[-1]),'jpeg')
    except:
        print(os.path.join(f'{data_root2}\\차종외관인식',i['json_data_id']))

end = time.time()
print("Time elapsed: ", timedelta(seconds=end-start))
count



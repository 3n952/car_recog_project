# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np 
import pandas as pd
import time 
import shutil 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 



current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
relative_parent_dir = os.path.join(current_dir, '..')
parent_dir = os.path.abspath(relative_parent_dir)



classid = pd.read_csv('classid.csv',index_col=0)
#classid = classid[classid['class_id']!='car-02'] 
classid = classid[classid['class_id']!='car-04']
classid = classid[classid['class_id']!='car-05']
classid = classid[classid['class_id']!='car-06']
classid['find'] = classid['label_'].apply(lambda x:x.split('#')[1])
classid = classid[classid['find'].str.contains('unknown', case=False)]

path = '../datasets/custom'

for i in classid['label_']: 
    try:
        shutil.rmtree((os.path.join(path,i)))
        print('remove')
    except: 
        pass

print('------------')
try:
    os.mkdir(os.path.join(parent_dir, 'datasets/custom'))
except: 
    print('already exist') 
    


folder_list = sorted(os.listdir(os.path.join(parent_dir, 'datasets/custom'))) 
path = os.path.join(parent_dir, 'datasets/custom')
for i in folder_list: 
    try:
        shutil.rmtree((os.path.join(path,i,'.ipynb_checkpoints')))
        print('remove')
    except: 
        #print('pass')
        pass

df = pd.DataFrame()
        
#path = r'..\datasets\custom'
for i in folder_list: 
    df_ = pd.DataFrame({'path': os.listdir(os.path.join(path,i)),'folder':i,'label_':i.split('.')[0]}) 
    #df = df.append(df_)
    df = pd.concat([df, df_], ignore_index=True)
    
    
# 10개 미만 라벨링 이미지 묶음 삭제
# need_delete_label = df.label_.value_counts()[df.label_.value_counts()<10].keys()
# df[df['label_'].isin(need_delete_label)].to_excel('..\\10개미만_제외차종목록.xlsx') #encoding='cp949'인자 제거
# df = df[~df['label_'].isin(need_delete_label)]
    
label = LabelEncoder()
label.fit(df['label_'])

df['label'] = label.transform(df['label_'])
df['path'] = df['folder']+'\\'+df['path']
df['path'] = df['path'].apply(lambda x: os.path.join(path,x))
df = df.reset_index(drop=True)
df = df.drop('folder',axis=1)

df = df.sample(df.shape[0])

df.drop_duplicates('label').sort_values('label').reset_index(drop=True).drop('path',axis=1).to_csv(r'..\label_encoding.csv') 

# 10개 미만 라벨링 삭제했을 때
# train_x,test_x_,train_y,test_y_ = train_test_split(df,df['label'],stratify=df['label'],test_size=0.2,random_state=22)
# val_x,test_x,val_y,test_y = train_test_split(test_x_,test_y_,stratify=test_y_,test_size=0.5,random_state=22) 

# 안했을 때
train_x,test_x_,train_y,test_y_ = train_test_split(df,df['label'],test_size=0.2,random_state=22)
val_x,test_x,val_y,test_y = train_test_split(test_x_,test_y_,test_size=0.5,random_state=22)

train_x.to_csv(os.path.join(parent_dir, 'train_x.csv'))
val_x.to_csv(os.path.join(parent_dir, 'val_x.csv'))
test_x.to_csv(os.path.join(parent_dir, 'test_x.csv'))

import os, sys, cv2, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from glob import glob


#각 feature에 대한 값 측정 및 bar그래프 시각화 
class cctv_EDA():
    def __init__(self,root_dir, is_train = True):
        self.root_dir = root_dir
        self.is_train = is_train
    

    # 추출시간(extract time)을 오전(a) 오후(b) 저녁(c) 심야(d)으로 나누기
    # input -> 06:00:10 output -> a
    def categorical_func(self, list_file):
        a = [7,8,9,10,11,12]
        b = [13,14,15,16,17,18]
        c = [19,20,21,22,23,24]
        d = [1,2,3,4,5,6]

        hh, __, __ = map(int, list_file.split(':'))

        if hh in a:
            return('a')
        elif hh in b:
            return('b')
        elif hh in c:
            return('c')
        else:
            return('d')
        

    # Label data(json)파일을 pandas.dataframe으로 변환 및 반환
    def json2df(self):
        allfiles = []

        if self.is_train:
            train_dir = os.path.join(self.root_dir, 'Training\\annotations')
            tfiles = glob(os.path.join(train_dir, '*.json'))
            
            for file in tfiles:
                with open(file, 'rb') as jsfile:
                    json_data = json.load(jsfile)
                    allfiles.append(json_data)
        
        else:
            val_dir = os.path.join(self.root_dir, 'Validation\\annotations')
            vfiles = glob(os.path.join(val_dir, '*.json'))
            
            for file in vfiles:
                with open(file, 'rb') as jsfile:
                    json_data = json.load(jsfile)
                    allfiles.append(json_data)

        print(len(allfiles),'files are prepared to convert(json to csv)')

        fname_list = []
        class_list = []
        weather_list = []
        timeline_list = []

        for i in range(len(allfiles)):
            fname_list.append(allfiles[i]['Source Data Info']['source_data_id'])
            class_list.append(allfiles[i]['Learning Data Info']['annotations'][0]['class_id'])
            weather_list.append(allfiles[i]['Raw Data Info']['weather'])
            timeline_list.append(self.categorical_func(allfiles[i]['Source Data Info']['extract_time']))
        
        cmr_df = pd.DataFrame(zip(fname_list, class_list, weather_list, timeline_list), columns=['file_name','class_num','weather','day_type'])
        print('----------------------------')
        print('success')

        return cmr_df
    
    # df 데이터 형식을 csv 파일로 변환 및 저장
    def save2csv(self, df):
        fname = input('csv file name')
        df.to_csv(fname, encoding = 'utf-8')

    
    # label 구성요소 시각화
    def count_visualize(self, csvfile):
        
        class_num = csvfile['class_num'].value_counts()
        weather_num = csvfile['weather'].value_counts()
        time_num = csvfile['day_type'].value_counts()

        fig, axes = plt.subplots(3,1,figsize = (10,8))
        sns.countplot(x = 'class_num', data = csvfile, ax = axes[0], order = csvfile['class_num'].value_counts().index)
        print('<table>', class_num)
        print()
        sns.countplot(x = 'weather', data = csvfile, ax = axes[1], order = csvfile['weather'].value_counts().index)
        print('<table>', weather_num)
        print()
        sns.countplot(x = 'day_type', data = csvfile, ax = axes[2], order = csvfile['day_type'].value_counts().index)
        print('<table>', time_num)
        print()

        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    print('ex. ''root/Training/annotations'' ')
    root_dir = input('input your root directory:')
    # root/Training/annotations와 같은 디렉토리 구조여야 한다.
    files = cctv_EDA(root_dir,False)
    dffiles = files.json2df()
    
    files.count_visualize(dffiles)
    



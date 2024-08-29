import os, sys, json
import pandas as pd
from glob import glob
from pathlib import Path
import ast

# txt 라벨을 정수 라벨링 + bbox 추출하기 위한 데이터 프레임화
def json2df(json_dir):
    df = pd.DataFrame(columns=['file_name','car_class', 'x', 'y', 'width', 'height'])

    fname_list = []
    with open(json_dir, 'rb') as jsfile:
        json_data = json.load(jsfile)
        fname_list.append(json_data['Source Data Info']['source_data_id'])

        for i in range(len(json_data['Learning Data Info']['annotations'])):
            class_list = []
            x_list = []
            y_list = []
            width_list = []
            height_list = []


            class_list.append(json_data['Learning Data Info']['annotations'][i]['model_id'])
            bbox_list = json_data['Learning Data Info']['annotations'][i]['coord']

            # coord 요소가 문자열인 경우에 숫자 리스트로 변환
            if isinstance(bbox_list, str):
                bbox_list = ast.literal_eval(bbox_list)

            x_list.append(bbox_list[0])
            y_list.append(bbox_list[1])
            width_list.append(bbox_list[2])
            height_list.append(bbox_list[3])

            new_df = pd.DataFrame(zip(fname_list, class_list, x_list, y_list, width_list, height_list), columns = ['file_name','car_class', 'x', 'y', 'width', 'height'])
            df = pd.concat([df, new_df], ignore_index= True)
    
    return df

# 데이터프레임의 car class를 string에서 int로 변환(모델 학습을 위해)
def string2int(df):
    df['label'], uniques = pd.factorize(df['car_class'])
    return df, uniques

def string2zero(df):
    df, _ = string2int(df)
    df['label'] = 0
    return df

# json2txt yolo annotation label
class json2yololabel():
    def __init__(self, json_dir, label_dir):
        self.json_dir = json_dir
        self.label_dir = label_dir

    def show_df(self, df):

        print('------원래 DataFrame------')
        print(df)

        df, uniques = string2int(df)
        print('-----변환된 DataFrame-----')
        print(df)
        print()
        print('label list:', uniques)
    
    def yololabeling_interclass(self):
        width = float(3840.)
        height = float(2160.)

        for i in range(len(os.listdir(self.json_dir))):
            df, _ = string2int(json2df(glob(os.path.join(self.json_dir, '*.json'))[i]))

            for imgname, contents in df.groupby(['file_name']):
                fname = os.path.join(self.label_dir, imgname[0]+'.txt')
                
                with open(fname, 'w') as anno:
                    for _, row in contents.iterrows():

                         # center x, center y, width, height  -> normalization
                        cx = round((row['x'] + (row['x'] + row['width'])) / 2 / width, 6)
                        cy = round((row['y'] + (row['y'] + row['height'])) / 2 / height, 6)

                        cwidth = round(float(row['width']) / width, 6)
                        cheight = round(float(row['height']) / height, 6)
                        
                        anno.write(f'{row["label"]} {cx} {cy} {cwidth} {cheight}\n')
    
    def yololabeling_justcar(self):
        width = float(3840)
        height = float(2160)

        for i in range(len(os.listdir(self.json_dir))):
            df = string2zero(json2df(glob(os.path.join(self.json_dir, '*.json'))[i]))

            for imgname, contents in df.groupby(['file_name']):
                fname = os.path.join(self.label_dir, imgname[0]+'.txt')
                
                with open(fname, 'w') as anno2:
                    for _, row in contents.iterrows():

                        # center x, center y, width, height  -> normalization
                        cx = round((row['x'] + (row['x'] + row['width'])) / 2 / width, 6)
                        cy = round((row['y'] + (row['y'] + row['height'])) / 2 / height, 6)

                        cwidth = round(float(row['width']) / width, 6)
                        cheight = round(float(row['height']) / height, 6)
                        anno2.write(f'{row["label"]} {cx} {cy} {cwidth} {cheight}\n')
                
if __name__ == '__main__':

    root_dir = r'C:\Users\QBIC\Desktop'

    tanno_dir = root_dir + '\\Training\\annotations'
    vanno_dir = root_dir + '\\Validation\\annotations'
    test_anno_dir = root_dir + '\\Test\\annotations'

    tlabel_dir = root_dir + '\\Training\\labels'
    vlabel_dir = root_dir + '\\Validation\\labels'
    test_label_dir = root_dir + '\\Test\\labels'

    try:
        os.makedirs(tlabel_dir, exist_ok=False)
        os.makedirs(vlabel_dir, exist_ok=False)
        os.makedirs(test_label_dir, exist_ok=False)
        print('mkdir success')
    except FileExistsError:
        print('already exist(label directory)')

    check = input('train mode: input"t"\nvalidation mode: input"v"\n')


        # label for train dataset
    if check =='t': #train
        tlabeling = json2yololabel(tanno_dir, tlabel_dir)

        # justcar => 첫번째줄 실행, intercar => 두번째줄 실행
        tlabeling.yololabeling_justcar()
        #tlabeling.yololabeling()
        print('complete labeling to txt format for train set')
    elif check =='v': #validation
        # label for val dataset
        vlabeling = json2yololabel(vanno_dir, vlabel_dir)

        # justcar => 첫번째줄 실행, intercar => 두번째줄 실행
        vlabeling.yololabeling_justcar()
        #vlabeling.yololabeling()
        print('complete labeling to txt format for validation set')
    
    elif check =="test":
        testlabeling = json2yololabel(test_anno_dir, test_label_dir)

        # justcar => 첫번째줄 실행, intercar => 두번째줄 실행
        testlabeling.yololabeling_justcar()
        #tlabeling.yololabeling()
        print('complete labeling to txt format for test set')

    else:
        print('check yout input.')
        check = input('train mode: input"t"\nvalidation mode: input"v"\n')


   






    







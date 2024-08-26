import pandas as pd
import os
from glob import glob
import shutil

# 기존 train_x.csv ..등등에 섞인 데이터에 unknown이 섞여있으면 포함시키지 않음. + 해당 이미지 삭제


def csv_filter(og_csv, new_csv, img_dir):
    # CSV 파일을 DataFrame으로 불러오기
    df = pd.read_csv(og_csv)

    column_name = 'label_'
    df['find'] = df[column_name].apply(lambda x:x.split('#')[1])

    # 'unknown'이 포함된 행을 삭제 + 해당 경로 이미지 파일 삭제
    df_copy = pd.DataFrame(columns=['find', 'path'])
    df_copy['find'] = df[column_name].apply(lambda x:x.split('#')[1])
    df_copy['path'] = df['path']

    df = df[~df['find'].str.contains('unknown', case = False)]

    # 변경된 DataFrame을 다시 CSV 파일로 저장 - 새로운 라벨링
    df.to_csv(new_csv, index=False) 

    idir = glob(os.path.join(img_dir, 'custom','*','*.jpg'))

    for _ in range(len(df)):
        for imgpath in df['path']:
            if not imgpath in idir:
                try:
                    os.remove(imgpath)
                    print(f"{imgpath} 파일이 삭제되었습니다.")

                except FileNotFoundError as e:
                    print(f"{imgpath} 파일이 없습니다.")

    
    # 디렉토리가 비어 있는지 확인 후 삭제

    dirpath = glob(os.path.join(img_dir, 'custom','*'))

    for dir_path in dirpath:
        try:
            if os.path.exists(dir_path) and len(os.listdir(dir_path)) == 0:
                os.rmdir(dir_path)
                print(f"{dir_path} 디렉토리가 삭제되었습니다.")

        except OSError as e:
            print(f"디렉토리를 삭제할 수 없습니다: {e}")


# img_dir = r'C:\Users\QBIC\Desktop\workspace\car_recog_project\transFG\datasets'

# for i in glob(os.path.join(r'C:\Users\QBIC\Desktop\workspace\car_recog_project\transFG', '*_x.csv')):
#     csv_filter(i, i[:-4] + '_new.csv', img_dir)


if __name__ == "__main__":

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    relative_parent_dir = os.path.join(current_dir, '..')
    parent_dir = os.path.abspath(relative_parent_dir)

    img_dir = r'C:\Users\QBIC\Desktop\workspace\car_recog_project\transFG\datasets'

    for i in glob(os.path.join(parent_dir, '*_x.csv')):
        csv_filter(i, i[:-4] + '.csv', img_dir)
    
    
    

       
     
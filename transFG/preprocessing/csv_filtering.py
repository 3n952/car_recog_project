import pandas as pd
import os

label_csv = r"..\label_encoding.csv"
# val_csv = r"..\val_x.csv"
# test_csv = r"..\test_x.csv"

new_label_csv = r"..\new_label_encoding.csv"
# new_val_csv = r"..\new_val_x.csv"
# new_test_csv = r"..\new_test_x.csv"

def csv_filter(og_csv, new_csv):
    # CSV 파일을 DataFrame으로 불러오기
    df = pd.read_csv(og_csv)

    # 특정 열의 이름 지정 (예: 'column_name')
    column_name = 'label_'

    # 'unknown'이 포함된 행을 삭제
    df = df[~df[column_name].str.contains('unknown', case=False, na=False)]
            
    # 정수형 label링 올림차순
    df_sorted = df.sort_values(by='label', ascending=True)

    # 삭제 시 불연속적인 정수 값을 채우기
    previous_value = -1

    for i in range(len(df_sorted)):
        current_value = df_sorted.iloc[i]['label']
        if current_value != previous_value + 1:
            df_sorted.at[df_sorted.index[i], 'label'] = previous_value + 1
        previous_value = df_sorted.iloc[i]['label']

    # 변경된 DataFrame을 다시 CSV 파일로 저장 (필요시)
    df_sorted.to_csv(new_csv, index=False)

if __name__ == "__main__":

    csv_filter(label_csv, new_label_csv)
    

       
     
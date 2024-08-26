import pandas as pd


def make_csv_custom(mode = 0):
    if mode == 0:
        #train mode
        # CSV 파일 읽기
        a_df = pd.read_csv('../train_x.csv')
        b_df = pd.read_csv('../custom_label_encoding.csv')

        # 'label_' 컬럼을 기준으로 조인하여 b_df의 'label' 값을 a_df에 추가
        a_df = a_df.merge(b_df[['label_', 'label']], on='label_', how='left', suffixes=('', '_test'))

        # 조인 후 a_df의 'label' 컬럼에 b_df의 'label' 값이 들어가도록 설정
        a_df['label'] = a_df['label_test']

        # # 불필요한 'label_b' 컬럼 제거
        a_df = a_df.drop(columns=['label_test'])

        # out range labeling 삭제
        a_df = a_df[a_df['label'] != 436]


        # # 변경된 DataFrame을 다시 CSV 파일로 저장
        a_df.to_csv('../train_x.csv', index=False)
        
    elif mode == 1:
        #val mode
        a_df = pd.read_csv('../val_x.csv')
        b_df = pd.read_csv('../custom_label_encoding.csv')

        a_df = a_df.merge(b_df[['label_', 'label']], on='label_', how='left', suffixes=('', '_test'))

        # 조인 후 a_df의 'label' 컬럼에 b_df의 'label' 값이 들어가도록 설정
        a_df['label'] = a_df['label_test']

        # # 불필요한 'label_b' 컬럼 제거
        a_df = a_df.drop(columns=['label_test'])

         # out range labeling 삭제
        a_df = a_df[a_df['label'] != 436]

        # # 변경된 DataFrame을 다시 CSV 파일로 저장
        a_df.to_csv('../val_x.csv', index=False)
    else:
        #test mode
        a_df = pd.read_csv('../test_x.csv')
        b_df = pd.read_csv('../custom_label_encoding.csv')

        # 'label_' 컬럼을 기준으로 조인하여 b_df의 'label' 값을 a_df에 추가
        a_df = a_df.merge(b_df[['label_', 'label']], on='label_', how='left', suffixes=('', '_test'))

        # 조인 후 a_df의 'label' 컬럼에 b_df의 'label' 값이 들어가도록 설정
        a_df['label'] = a_df['label_test']

        # # 불필요한 'label_b' 컬럼 제거
        a_df = a_df.drop(columns=['label_test'])

         # out range labeling 삭제
        a_df = a_df[a_df['label'] != 436]

        # # 변경된 DataFrame을 다시 CSV 파일로 저장
        a_df.to_csv('../test_x.csv', index=False)

    print("label_encoding.csv 파일이 성공적으로 업데이트되었습니다.")




# mode = 0 or 1 or ...
make_csv_custom(mode = 2)
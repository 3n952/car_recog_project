import cv2
import os
import json
from os.path import join
import os
import numpy as np
import pandas as pd
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg


class custom_dataloader():
    def __init__(self, root, dtype=0,  data_len=None, transform=None, remove_unknown=False):
        self.root = root
        self.dtype = dtype
        self.transform = transform

        if remove_unknown:
            if self.dtype ==2: 
                self.test_x = pd.read_csv('test_x_new.csv')
            else:    
                self.train_x = pd.read_csv('train_x_new.csv')
                self.val_x = pd.read_csv('val_x_new.csv')
        
        else:
            if self.dtype ==2: 
                self.test_x = pd.read_csv('test_x.csv')
            else:    
                self.train_x = pd.read_csv('train_x.csv')
                self.val_x = pd.read_csv('val_x.csv')
        
    def __getitem__(self, index): 
        
        if self.dtype ==0:
        
            img = cv2.imread(self.train_x['path'].iloc[index])
            
            # file명에 로마숫자 들어간 경우 이미지를 로드 못함. -> 전처리하거나 못읽는 이미지 예외처리 하기
            if img is not None:
                target =  self.train_x['label'].iloc[index]

                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode = "RGB")
                if self.transform is not None:
                    img = self.transform(img)

            else:
                # 이미지 로드 실패 시 처리 (예: 예외 발생, 기본 이미지 반환 등)
                img = Image.new("RGB", (100, 100), (0, 0, 0))  # 예시: 빈 이미지 생성
                target = 203  # 기본값 설정 - label_encoding.csv 파일에서 마지막 라벨 하나 추가하는 역할

        elif self.dtype == 1:
            img = cv2.imread(self.val_x['path'].iloc[index]) 
            
            if img is not None:
                target =  self.val_x['label'].iloc[index]

                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode = "RGB")
                if self.transform is not None:
                    img = self.transform(img)
            else:
                # 이미지 로드 실패 시 처리 (예: 예외 발생, 기본 이미지 반환 등)
                img = Image.new("RGB", (100, 100), (0, 0, 0))  # 예시: 빈 이미지 생성
                target = 203  # 기본값 (예: -1, None 등) 설정


        elif self.dtype ==2: 
            #print(self.test_x['path'].iloc[index])
            img = cv2.imread(self.test_x['path'].iloc[index])
            
            
            if img is not None:
                target =  self.test_x['label'].iloc[index]

                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode = "RGB")
                
                if self.transform is not None:
                    img = self.transform(img)
            else:
                # 이미지 로드 실패 시 처리 (예: 예외 발생, 기본 이미지 반환 등)
                img = Image.new("RGB", (100, 100), (0, 0, 0))  # 예시: 빈 이미지 생성
                target = 203  # 기본값 (예: -1, None 등) 설정
        

        return img, target

    def __len__(self):
        if self.dtype == 0:
            return len(self.train_x)
        elif self.dtype == 1:
            return len(self.val_x) 
        elif self.dtype == 2:
            return len(self.test_x) 



import multiprocessing

if __name__ == "___main__":
    multiprocessing.freeze_support()
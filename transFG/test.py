from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

import multiprocessing
import logging
import argparse
import os 
import cv2
import pandas as pd
import random
import numpy as np
import time
from PIL import Image
from datetime import timedelta

import sys
sys.path.append('../')
from models.modeling import VisionTransformer, CONFIGS
import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_utils import get_world_size
from utils.dataset import custom_dataloader
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=20, type=int,
                        help="Total batch size for training.")  
args = parser.parse_args() 
batch_size = args.batch_size

test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                            transforms.CenterCrop((448, 448)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# test set 구성
testset = custom_dataloader(root='datasets/custom', dtype=2, transform = test_transform) 

test_sampler = SequentialSampler(testset)#if args.local_rank == -1 else DistributedSampler(testset) #SequentialSampler : 항상 같은 순서
test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=batch_size,
                             num_workers=0,
                             pin_memory=True) if testset is not None else None 

img_size = 448
smoothing_value = 0.0
pretrained_model = "output/test_checkpoint.bin"
config = CONFIGS["ViT-B_16"]
config.split = 'overlap'
config.slide_step = 12
num_classes = pd.read_csv('custom_label_encoding.csv')['label'].nunique()
print(num_classes,'num_classes')
model = VisionTransformer(config, img_size, 436, smoothing_value, zero_head=True) 

if pretrained_model is not None:
    pretrained_model = torch.load(pretrained_model, map_location=torch.device('cpu'))['model']
    model.load_state_dict(pretrained_model) 
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

epoch_iterator = tqdm(test_loader) 
all_preds, all_label = [], [] 
with torch.no_grad():
    for step, batch in enumerate(epoch_iterator): 
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        if len(y)==1:
            print(y)
            break
        loss, logits = model(x, y)
        preds = torch.argmax(logits, dim=-1) 
        all_label.append(list(y.cpu().numpy()))
        all_preds.append(list(preds.cpu().numpy())) 

        
all_preds = np.array(sum(all_preds,[])) # 차원을 하나 없에줌
all_label = np.array(sum(all_label,[]))

test = pd.read_csv('test_x.csv',index_col=0)
test['추론'] = all_preds 
test.rename({'label':'정답','label_':'정답_차량이름'},axis=1) 
test.to_csv('results/all_result.csv')

print('Accuracy :',accuracy_score(all_label, all_preds)) 
print('Precision :',precision_score(all_label, all_preds, average='weighted')) 
print('Recall :',recall_score(all_label, all_preds, average='weighted')) 
print('F1 score :',f1_score(all_label, all_preds, average='weighted'))

pd.DataFrame({'Accuracy' :accuracy_score(all_label, all_preds), 
              'Precision' :precision_score(all_label, all_preds, average='weighted'), 
              'Recall':recall_score(all_label, all_preds, average='weighted'), 
              'F1 score':f1_score(all_label, all_preds, average='weighted')
             },[0]).to_csv('results/classification_result.csv')

def report_to_df(report):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+[x for x in report[0] if x!='']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value!='']
        if row!=[]:
            values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    return df

if __name__ == "___main__":
    multiprocessing.freeze_support()
    df = report_to_df(classification_report(all_label, all_preds))
    df_ = pd.read_csv('label_encoding.csv',index_col=0).rename({'label':'Class Name'},axis=1) 
    df_['Class Name'] = df_['Class Name'].astype('str') 
    df = pd.merge(df,df_, how='left').drop_duplicates('Class Name').reset_index(drop=True)[['label_','Class Name','precision','recall','f1-score','support']]
    df.to_excel('results/classification_report.xlsx')


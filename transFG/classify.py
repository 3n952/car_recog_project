from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from datetime import timedelta
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from models.modeling import VisionTransformer, CONFIGS
import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_utils import get_world_size
from utils.dataset import custom_dataloader
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=16, type=int,
                        help="Total batch size for training.")  
args = parser.parse_args() 
batch_size = args.batch_size

def unnormalize(tensor, mean, std):
    #transform 적용한 부분을 denormalizing.

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

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
refer = pd.read_csv('custom_label_encoding.csv', index_col=0)
num_classes = refer['label'].nunique()
print(num_classes,': num_classes to classify')

model = VisionTransformer(config, img_size, 436, smoothing_value, zero_head=True) 

if pretrained_model is not None:
    pretrained_model = torch.load(pretrained_model, map_location=torch.device('cpu'))['model']
    model.load_state_dict(pretrained_model) 
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

epoch_iterator = test_loader
 
with torch.no_grad():
    for step, batch in enumerate(epoch_iterator):
        print(f"######################### {step + 1} 번째 batch #########################") 
        batch = tuple(t.to(device) for t in batch)
        x, y = batch

        if len(y) == 1:
            print(y)
            break

        _, logits = model(x, y)
        preds = torch.argmax(logits, dim=-1)

        #tensor([1,3,4,1,5,2]) 꼴 -> 텐서 인덱스 접근 
        #print(f"{preds[step]} 예측값, {y[step]} 정답값")
        # 이미지 배치를 반복하여 각 이미지를 시각화
        for i in range(x.shape[0]):
            # preds , y per image
            preds_ = preds[i]
            y_ = y[i]

            #string value
            pred_val = refer.loc[refer['label'] == preds_.numpy(), 'label_'].values
            y_val = refer.loc[refer['label'] == y_.numpy(), 'label_'].values

            # Unnormalize the image
            unnormalized_image = unnormalize(x.clone()[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            # Convert tensor to numpy array and transpose to (H, W, C)
            image = unnormalized_image.numpy().transpose(1, 2, 0)

            # Clip the values to be in the range [0, 1] for visualization
            #unnormalized_image = np.clip(unnormalized_image, 0, 1)

            print(f"label: {y_val} -------- pred: {pred_val}")
            
            # 이미지 시각화
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.axis('off')
            plt.show()




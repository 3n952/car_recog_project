import torch
import pandas as pd
import sys
sys.path.append('models')
from modeling import VisionTransformer, CONFIGS
from torch.nn.modules.utils import _pair

img_size = 448
smoothing_value = 0.0
config = CONFIGS["ViT-B_16"]

patch_size = _pair(config.patches["size"])

config.split = 'overlap'
num_classes = pd.read_csv('../label_encoding.csv')['label'].nunique()

model_path = "../output/test_checkpoint.bin"
# model load
vit_model = VisionTransformer(config, img_size, num_classes, smoothing_value, zero_head=True) 

# model = torch.load(model_path, map_location=torch.device('cpu'))['model']
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
#vit_model.load_state_dict(model, strict=False) 


# 체크포인트의 키 확인
print("model dict key: ", model.keys())
# print()
# print('model["model"] key: ', model['model'].keys())
# print()
# print('model["amp"] key: ', model['amp'].keys())

# state_dict 확인
state_dict = model['model']

for name, param in state_dict.items():
    print(f"Layer: {name} | Shape: {param.shape}")
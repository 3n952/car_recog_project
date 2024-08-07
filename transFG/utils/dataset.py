import os
import json

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

# from torchvision.datasets import VisionDataset
# from torchvision.datasets.folder import default_loader
# from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg


# car model recog dataloader
class CarDataset2(Dataset):

    def __init__(self, json_anno, image_dir, car_names, cleaned=None, transform=None):
        """
        Args:
            json_anno (string): Path to the json annotation file.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.json_dataset = os.path.join(json_anno)

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.data_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name).convert('RGB')
        car_class = self.car_annotations[idx][-2][0][0]
        car_class = torch.from_numpy(np.array(car_class.astype(np.float32))).long() - 1
        assert car_class < 196
        
        if self.transform:
            image = self.transform(image)

        # return image, car_class, img_name
        return image, car_class

    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret

    def show_batch(self, img_batch, class_batch):

        for i in range(img_batch.shape[0]):
            ax = plt.subplot(1, img_batch.shape[0], i + 1)
            title_str = self.map_class(int(class_batch[i]))
            img = np.transpose(img_batch[i, ...], (1, 2, 0))
            ax.imshow(img)
            ax.set_title(title_str.__str__(), {'fontsize': 5})
            plt.tight_layout()

def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images
    
def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


class CarDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, 'images'))))
        self.labels = list(sorted(os.listdir(os.path.join(root_dir, 'labels'))))

    def __getitem__(self, idx):
        # 이미지, 라벨 path load
        img_path = os.path.join(self.root_dir, 'images', self.imgs[idx])
        label_path = os.path.join(self.root_dir, 'labels', self.labels[idx])

        # img, label load 및 할당
        img = read_image(img_path)

        with open(label_path, 'rb') as jsfile:
            label = json.load(jsfile)
        
        




        



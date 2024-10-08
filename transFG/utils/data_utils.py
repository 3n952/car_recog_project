import logging
from PIL import Image
import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import custom_dataloader
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'custom':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
        val_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])                                    
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        print(args.data_root)
        trainset = custom_dataloader(root=args.data_root, dtype=0, transform = train_transform, remove_unknown=False)
        valset = custom_dataloader(root=args.data_root, dtype=1, transform = val_transform, remove_unknown=False)
        testset = custom_dataloader(root=args.data_root, dtype=2, transform = test_transform, remove_unknown=False) 
    
    else:
        print('build new get_loader function')

    if args.local_rank == 0:
        torch.distributed.barrier()
        
    #RandomSampler : 데이터셋을 적절히 섞음
    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset) #RandomSampler : 랜덤
    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset) #SequentialSampler : 항상 같은 순서
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=0,
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                             #sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=True) if valset is not None else None
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=True) if testset is not None else None

    return train_loader, val_loader, test_loader

# -*- coding: utf-8 -*-
import os
import torch
from torchvision import transforms
from .datasets import register_dataset
from module import utils


@register_dataset('DomainNet')
class DomainNetDataset:
    """
    DomainNet Dataset class
    """
    def __init__(self, name, img_dir, LDS_type, is_target):
        self.name = name
        self.img_dir = img_dir
        self.LDS_type = LDS_type
        self.is_target = is_target

    def get_data(self):
        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize_transform
        ])

        train_path = os.path.join('data/DomainNet/', '{}_{}.txt'.format(self.name, 'list'))
        test_path = os.path.join('data/DomainNet/', '{}_{}.txt'.format(self.name, 'list'))

        train_dataset = utils.ImageList(open(train_path).readlines(), os.path.join(self.img_dir, 'images'))
        val_dataset = utils.ImageList(open(test_path).readlines(), os.path.join(self.img_dir, 'images'))
        test_dataset = utils.ImageList(open(test_path).readlines(), os.path.join(self.img_dir, 'images'))
        self.num_classes = 126

        train_dataset.targets, val_dataset.targets, test_dataset.targets = torch.from_numpy(train_dataset.labels), \
                                                                           torch.from_numpy(val_dataset.labels), \
                                                                           torch.from_numpy(test_dataset.labels)
        return self.num_classes, train_dataset, val_dataset, test_dataset, self.train_transforms, self.test_transforms

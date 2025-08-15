# -*- coding: utf-8 -*-
import os
import random
import torch
from torchvision import transforms
from .datasets import register_dataset
from module import utils
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 2.0]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

@register_dataset('VisDA2017')
class VisDADataset:
	"""
	VisDA Dataset class
	"""

	def __init__(self, name, img_dir, LDS_type, is_target):
		self.name = name
		self.img_dir = img_dir
		self.LDS_type = LDS_type
		self.is_target = is_target

	def get_data(self):
		self.train_transforms = transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
			transforms.RandomApply(
				[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
				p=0.8,
			),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()
		])
		self.test_transforms = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.CenterCrop((224, 224)), # add 250318
				transforms.ToTensor(),
			])
		train_path = os.path.join('data/VisDA2017/', '{}.txt'.format(self.name))
		test_path = os.path.join('data/VisDA2017/', '{}.txt'.format(self.name))
		if self.name == 'visda_train':
			folder_name = 'train'
		else:
			folder_name = 'validation'
		train_dataset = utils.ImageList(open(train_path).readlines(), os.path.join(self.img_dir, 'images', folder_name))
		val_dataset = utils.ImageList(open(test_path).readlines(), os.path.join(self.img_dir, 'images', folder_name))
		test_dataset = utils.ImageList(open(test_path).readlines(), os.path.join(self.img_dir, 'images', folder_name))

		self.num_classes = 12
		if self.LDS_type == 'pda' and self.is_target:
			self.num_classes = 6
		train_dataset.targets, val_dataset.targets, test_dataset.targets = torch.from_numpy(train_dataset.labels), \
																		   torch.from_numpy(val_dataset.labels), \
																		   torch.from_numpy(test_dataset.labels)
		return self.num_classes, train_dataset, val_dataset, test_dataset, self.train_transforms, self.test_transforms


import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from dataloaders.helper import CutoutPIL
from randaugment import RandAugment
import xml.dom.minidom


class voc2007(data.Dataset):
    def __init__(self, root, data_split, img_size=224, p=1, annFile="", label_mask=None, partial=1+1e-6):
        # data_split = train / val
        self.root = root
        self.classnames = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']

        if annFile == "":
            self.annFile = os.path.join(self.root, 'Annotations')
        else:
            raise NotImplementedError

        image_list_file = os.path.join(self.root, 'ImageSets', 'Main', '%s.txt' % data_split)

        with open(image_list_file) as f:
            image_list = f.readlines()
        self.image_list = [a.strip() for a in image_list]

        self.data_split = data_split
        if data_split == 'Train':
            num_examples = len(self.image_list)
            pick_example = int(num_examples * p)
            self.image_list = self.image_list[:pick_example]
        else:
            self.image_list = self.image_list

        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(img_size)
            transforms.Resize((img_size, img_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        test_transform = transforms.Compose([
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        if self.data_split == 'trainval':
            self.transform = train_transform
        elif self.data_split == 'test':
            self.transform = test_transform
        else:
            raise ValueError('data split = %s is not supported in Nus Wide' % self.data_split)

        # create the label mask
        self.mask = None
        self.partial = partial
        if data_split == 'trainval' and partial < 1.:
            if label_mask is None:
                rand_tensor = torch.rand(len(self.image_list), len(self.classnames))
                mask = (rand_tensor < partial).long()
                mask = torch.stack([mask], dim=1)
                torch.save(mask, os.path.join(self.root, 'Annotations', 'partial_label_%.2f.pt' % partial))
            else:
                mask = torch.load(os.path.join(self.root, 'Annotations', label_mask))
            self.mask = mask.long()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'JPEGImages', self.image_list[index] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        ann_path = os.path.join(self.annFile, self.image_list[index] + '.xml')
        label_vector = torch.zeros(20)
        DOMTree = xml.dom.minidom.parse(ann_path)
        root = DOMTree.documentElement
        objects = root.getElementsByTagName('object')
        for obj in objects:
            if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                continue
            tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
            label_vector[self.classnames.index(tag)] = 1.0
        targets = label_vector.long()
        target = targets[None, ]
        if self.mask is not None:
            masked = - torch.ones((1, len(self.classnames)), dtype=torch.long)
            target = self.mask[index] * target + (1 - self.mask[index]) * masked

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def name(self):
        return 'voc2007'


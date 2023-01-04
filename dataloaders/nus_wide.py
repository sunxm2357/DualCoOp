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
import pickle

class NUSWIDE_ZSL(data.Dataset):
    def __init__(self, root, data_split, img_size=224, p=1, annFile="", label_mask=None, partial=1+1e-6):
        # data_split = train / val
        ann_file_names = {'train': 'formatted_train_all_labels_filtered.npy',
                          'val': 'formatted_val_all_labels_filtered.npy',
                          'val_gzsl': 'formatted_val_gzsl_labels_filtered_small.npy',
                          'test_gzsl': 'formatted_val_gzsl_labels_filtered.npy'}
        img_list_name = {'train': 'formatted_train_images_filtered.npy',
                         'val': 'formatted_val_images_filtered.npy',
                         'val_gzsl': 'formatted_val_gzsl_images_filtered_small.npy',
                         'test_gzsl': 'formatted_val_gzsl_images_filtered.npy'}
        self.root = root
        class_name_files = os.path.join(self.root, 'annotations', 'Tag_all', 'all_labels.txt')
        with open(class_name_files) as f:
            classnames = f.readlines()
        self.classnames = [a.strip() for a in classnames]

        if annFile == "":
            annFile = os.path.join(self.root, 'annotations', 'zsl', ann_file_names[data_split])
        else:
            raise NotImplementedError
        cls_id = pickle.load(open(os.path.join(self.root, 'annotations', 'zsl', "cls_id.pickle"), "rb"))
        if data_split == 'train':
            cls_id = cls_id['seen']
        elif data_split == 'val':
            cls_id = cls_id['unseen']
        elif data_split in ['val_gzsl', 'test_gzsl']:
            cls_id = list(range(1006))
        else:
            raise ValueError
        self.cls_id = cls_id
        image_list = os.path.join(self.root, 'annotations', 'zsl', img_list_name[data_split])
        self.anns = np.load(annFile)

        self.image_list = np.load(image_list)
        assert len(self.anns) == len(self.image_list)

        self.data_split = data_split
        ids = list(range(len(self.image_list)))
        if data_split == 'train':
            num_examples = len(ids)
            pick_example = int(num_examples * p)
            self.ids = ids[:pick_example]
        else:
            self.ids = ids

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

        if self.data_split == 'train':
            self.transform = train_transform
        elif self.data_split in ['val', 'val_gzsl', 'test_gzsl']:
            self.transform = test_transform
        else:
            raise ValueError('data split = %s is not supported in Nus Wide' % self.data_split)

        # create the label mask
        self.mask = None
        self.partial = partial
        if data_split == 'train' and partial < 1.:
            if label_mask is None:
                rand_tensor = torch.rand(len(self.ids), len(self.classnames))
                mask = (rand_tensor < partial).long()
                mask = torch.stack([mask], dim=1)
                torch.save(mask, os.path.join(self.root, 'annotations', 'partial_label_%.2f.pt' % partial))
            else:
                mask = torch.load(os.path.join(self.root, 'annotations', label_mask))
            self.mask = mask.long()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.root, 'images', self.image_list[img_id].strip())
        img = Image.open(img_path).convert('RGB')
        targets = self.anns[img_id]
        targets = torch.from_numpy(targets).long()
        target = targets[None, ]
        if self.mask is not None:
            masked = - torch.ones((1, len(self.classnames)), dtype=torch.long)
            target = self.mask[index] * target + (1 - self.mask[index]) * masked

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def name(self):
        return 'nus_wide'

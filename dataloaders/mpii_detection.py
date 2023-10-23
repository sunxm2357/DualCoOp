import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
#from torchvision import datasets as datasets
#from pycocotools.coco import COCO
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from dataloaders.helper import CutoutPIL
from randaugment import RandAugment
import pickle


class MPII_ZSL(data.Dataset):
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
    

class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, data_split, img_size=224, p=1, annFile="", label_mask=None, partial=1+1e-6):
        # super(CocoDetection, self).__init__()
        self.classnames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                           "kite",
                           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                           "orange",
                           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]
        self.root = root
        if annFile == "":
            annFile = os.path.join(self.root, 'annotations', 'instances_%s.json' % data_split)
            cls_id = list(range(len(self.classnames)))
        else:
            cls_id = pickle.load(open(os.path.join(self.root, 'annotations', "cls_ids.pickle"), "rb"))
            if 'train' in annFile:
                cls_id = cls_id["train"]
            elif "val" in annFile:
                if "unseen" in annFile:
                    cls_id = cls_id["test"]
                else:
                    cls_id = cls_id['train'] | cls_id['test']
            else:
                raise ValueError("unknown annFile")
            cls_id = list(cls_id)
        cls_id.sort()
        self.coco = COCO(annFile)
        self.data_split = data_split
        ids = list(self.coco.imgToAnns.keys())
        if data_split == 'train2014':
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

        if self.data_split == 'train2014':
            self.transform = train_transform
        elif self.data_split == "val2014":
            self.transform = test_transform
        else:
            raise ValueError('data split = %s is not supported in mscoco' % self.data_split)

        self.cat2cat = dict()
        cats_keys = [*self.coco.cats.keys()]
        cats_keys.sort()
        for cat, cat2 in zip(cats_keys, cls_id):
            self.cat2cat[cat] = cat2
        self.cls_id = cls_id

        # create the label mask
        self.mask = None
        self.partial = partial
        if data_split == 'train2014' and partial < 1.:
            if label_mask is None:
                rand_tensor = torch.rand(len(self.ids), len(self.classnames))
                mask = (rand_tensor < partial).long()
                mask = torch.stack([mask, mask, mask], dim=1)
                torch.save(mask, os.path.join(self.root, 'annotations', 'partial_label_%.2f.pt' % partial))
            else:
                mask = torch.load(os.path.join(self.root, 'annotations', label_mask))
            self.mask = mask.long()


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
        return 'mpii'










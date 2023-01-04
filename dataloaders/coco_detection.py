import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
from torchvision import datasets as datasets
from pycocotools.coco import COCO
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from dataloaders.helper import CutoutPIL
from randaugment import RandAugment
import pickle


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
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, len(self.classnames)), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output
        if self.mask is not None:
            masked = - torch.ones((3, len(self.classnames)), dtype=torch.long)
            target = self.mask[index] * target + (1 - self.mask[index]) * masked

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, self.data_split, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def name(self):
        return 'coco'

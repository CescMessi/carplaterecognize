
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import random
from findplate.config import opt


class MyDataset(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        
        if self.test:
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
        else:
            dataset = ImageFolder(root)
            self.data_classes = dataset.classes
            imgs = [dataset.imgs[i][0] for i in range(len(dataset.imgs))]
            labels = [dataset.imgs[i][1] for i in range(len(dataset.imgs))]
        imgs_num = len(imgs)
        
        if self.test:
            self.imgs = imgs

        # 按7:3的比例划分训练集和验证集
        elif train:
            self.imgs = []
            self.labels = []
            for i in range(imgs_num):
                if random.random()<0.7:
                    self.imgs.append(imgs[i])
                    self.labels.append(labels[i])
        else:
            self.imgs = []
            self.labels = []
            for i in range(imgs_num):
                if random.random()>0.7:
                    self.imgs.append(imgs[i])
                    self.labels.append(labels[i])
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def id_to_class(self, index):
        return self.data_classes(index)

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        if self.test:
            # label = self.imgs[index].split('.')[-2].split('/')[-1]
            label = img_path.split('/')[-1]
        else:
            label = self.labels[index]
        data = Image.open(img_path)
        if opt.gray == True:
            dataRGB = data.convert('RGB')
            dataRGB = self.transforms(dataRGB)
            return dataRGB, label
        
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

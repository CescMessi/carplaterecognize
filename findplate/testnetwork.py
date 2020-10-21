
from findplate.config import opt
import os
import sys
import torch as t
from findplate import models
from findplate.data.dataset import MyDataset
from torch.utils.data import DataLoader
from torchnet import meter
from findplate.utils.visualize import Visualizer
from tqdm import tqdm
from torchvision import transforms as T

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# 判断是否为车牌
@t.no_grad()
def detect(img):
    # 载入模型和参数
    model = getattr(models, opt.model)().eval()
    model.load(resource_path('findplate/checkpoints/squeezenet_plate.pth'))
    # 归一化
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    # 变换
    transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
    inputdata = transforms(img)
    inputdata = t.unsqueeze(inputdata, 0)
    # 将图像喂入模型，获取标签
    score = model(inputdata)
    id_label_dict = row_csv2dict(resource_path('findplate/plate.csv'))
    label = score.max(dim = 1)[1].detach().tolist()
    label = [id_label_dict[str(i)] for i in label]
    return label

# 识别字符
@t.no_grad()
def identify(img_array):
    model = getattr(models, 'SqueezeNetGray')().eval()
    model.load(resource_path('findplate/checkpoints/squeezenet_char.pth'))
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])

    # 将多个图像合并到一个tensor中
    flag = 0
    for img in img_array:
        data = transforms(img)
        data = t.unsqueeze(data, 0)
        if flag == 0:
            inputdata = data
            flag = 1
        else:
            inputdata = t.cat((inputdata, data), 0)
    
    
    score = model(inputdata)
    id_label_dict = row_csv2dict(resource_path('findplate/char.csv'))
    label = score.max(dim = 1)[1].detach().tolist()
    label = [id_label_dict[str(i)] for i in label]
    return label




def row_csv2dict(csv_file):
    import csv
    dict_club={}
    with open(csv_file)as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            dict_club[row[0]]=row[1]
    return dict_club
    

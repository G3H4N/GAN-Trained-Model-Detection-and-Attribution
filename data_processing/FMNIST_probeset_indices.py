# coding:utf-8
"""
DataSetSegmentation.py
==========================
Segment a torchvision dataset.py for the training and testing of the target model and shadow structures.
"""

import torch
import torchvision
import pickle  # pickle提供了一个简单的持久化功能，可以序列化对象并以文件的形式存放在磁盘上
import os
import PIL.Image as Image


def saveData(dir, filename, obj):

    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir + filename):
        # 注意字符串中含有空格，所以有r' '；touch指令创建新的空文件
        os.system(r'touch {}'.format(dir + filename))

    with open(dir + filename, 'wb') as file:
        # pickle.dump(obj,file[,protocol])，将obj对象序列化存入已经打开的file中，file必须以二进制可写模式打开（“wb”），可选参数protocol表示高职pickler使用的协议，支持的协议有0，1，2，3，默认的协议是添加在python3中的协议3。
        pickle.dump(obj, file)
        file.close()
    print('Save data', dir + filename, '......ok')

    return


def loadData(filename):

    with open(filename, 'rb') as file:
        print('Open file', filename, '......ok')
        obj = pickle.load(file, encoding='latin1')
        file.close()
        return obj



transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(64, 64), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)

prob_size = 100
FMNIST = torchvision.datasets.FashionMNIST('/data/gehan/Datasets/FashionMNIST/', train=False, download=True, transform=transform)

index = torch.rand(prob_size) * len(FMNIST)
index = index.floor()
index = index.int()


if os.path.exists('/data/gehan/Datasets/FashionMNIST/' + 'ProbeSet_indices'):
    print('ProbeSet_indices already exists.')
else:
    saveData('/data/gehan/Datasets/FashionMNIST/', 'ProbeSet_indices', index)
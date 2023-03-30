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


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = '/data/gehan/Datasets/CelebA/png/align/Split_align_Attr_22_21_20'
sets_dir = '/data/gehan/Datasets/CelebA/png/align/CelebA_png_align_Attr_22_21_20_sets/'

CelebA = torchvision.datasets.ImageFolder(data_dir, transform=transform)
ClassNum = len(CelebA.classes)
'''
divide = [int(len(CelebA) * 0.5 * 2. / 3.),
          int(int(len(CelebA) * 0.5) - int(len(CelebA) * 0.5 * 2. / 3.)),
          int((len(CelebA) - int(len(CelebA) * 0.5)) * 2. / 3.),
          (len(CelebA) - int(len(CelebA) * 0.5)) - int((len(CelebA) - int(len(CelebA) * 0.5)) * 2. / 3.)]
'''
len_CelebA = len(CelebA)
size_probeset = 100
size_testset = 2499
size_trainset = len_CelebA - size_probeset - size_testset

divide = [size_probeset, size_testset, size_trainset]

print(divide, len(CelebA))
ProbeSet, TestSet, TrainSet = torch.utils.data.random_split(CelebA, divide)

saveData(sets_dir, 'ProbeSet', ProbeSet)
saveData(sets_dir, 'TestSet', TestSet)
saveData(sets_dir, 'TrainSet', TrainSet)

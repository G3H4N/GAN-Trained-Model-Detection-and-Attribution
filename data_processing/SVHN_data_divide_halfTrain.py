# coding:utf-8


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
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)

SVHN_train = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform)
SVHN_test = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform, split='test')

lenth_sub_train = int(len(SVHN_train)/2)
split = [lenth_sub_train, len(SVHN_train) - lenth_sub_train]
sub_train1, sub_train2 = torch.utils.data.random_split(SVHN_train, split)

saveData('/data/gehan/Datasets/SVHN/', 'SVHN_train_half1', sub_train1)
saveData('/data/gehan/Datasets/SVHN/', 'SVHN_train_half2', sub_train2)
